"""Location-only regions with explicit cannot-link constraints across broad ridges."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from pipeline_common import connect, read_view
from time_splits import digest
from train_grouped import save_json

ROOT = Path(__file__).resolve().parents[1]


def terrain_hashes(path):
    import rasterio
    with rasterio.open(path) as source:
        return {Path(f).name:digest(Path(f)) for f in source.files}


def broad_barrier(elevation, step_m, relief_m=300., width_m=3000., peak_m=500.):
    """A broad high barrier above BOTH endpoints, not merely a steep hillside."""
    if not np.isfinite(elevation).all():
        return None
    high = (elevation >= max(elevation[0], elevation[-1])+relief_m) & (elevation >= peak_m)
    best = current = 0
    for flag in high:
        current = current+1 if flag else 0
        best = max(best, current)
    return bool(max(0, best-1)*step_m >= width_m)


def partition(xy, blocked, k):
    """Hard pair constraints; no relaxation when a requested K is infeasible."""
    n = len(xy)
    if k < 2 or k > 4 or n < k or blocked.shape != (n,n):
        raise ValueError('Use K=2..4 and a square pair matrix')
    with threadpool_limits(limits=2):
        initial = KMeans(n_clusters=k, n_init=20, random_state=42).fit(xy)
    if not blocked.any():
        return initial.labels_, {'method': 'location KMeans', 'feasible': True}
    pairs = np.argwhere(np.triu(blocked, 1))
    a = lil_matrix((n+k+len(pairs)*k, n*k))
    lo = np.full(a.shape[0], -np.inf)
    hi = np.full(a.shape[0], np.inf)
    for i in range(n):
        a[i,i*k:(i+1)*k] = 1
        lo[i] = hi[i] = 1
    for g in range(k):
        a[n+g,g::k] = 1
        lo[n+g] = 1
    row = n+k
    for i,j in pairs:
        for g in range(k):
            a[row,i*k+g] = a[row,j*k+g] = 1
            hi[row] = 1
            row += 1
    centers = initial.cluster_centers_
    labels = None
    statuses = []
    for _ in range(5):
        costs = ((xy[:,None,:]-centers[None,:,:])**2).sum(axis=2).ravel()/1e6
        solution = milp(costs, integrality=np.ones(n*k), bounds=Bounds(0,1),
            constraints=LinearConstraint(a.tocsr(),lo,hi), options={'time_limit':60, 'mip_rel_gap':.01})
        statuses.append({'status':int(solution.status), 'message':solution.message})
        if solution.x is None:
            return None, {'feasible':False, 'solver':statuses,
                          'reason':'infeasible' if solution.status==2 else 'solver found no feasible assignment'}
        new = solution.x.reshape(n,k).argmax(axis=1)
        if len(set(new)) != k or any(new[i]==new[j] for i,j in pairs):
            raise ValueError('Solver assignment violates hard ridge constraints')
        if labels is not None and np.array_equal(labels,new):
            break
        labels = new
        centers = np.array([xy[labels==g].mean(axis=0) for g in range(k)])
    return labels, {'feasible':True, 'method':'location SSE with hard broad-ridge cannot-links', 'solver':statuses}


def prepare_dem(path):
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    with rasterio.open(path) as source:
        transform,width,height = calculate_default_transform(source.crs,'EPSG:5179',source.width,
            source.height,*source.bounds,resolution=1000)
        dem = np.full((height,width),np.nan,dtype='float32')
        reproject(rasterio.band(source,1),dem,src_transform=source.transform,src_crs=source.crs,
                  dst_transform=transform,dst_crs='EPSG:5179',src_nodata=source.nodata,dst_nodata=np.nan,
                  resampling=Resampling.average,num_threads=2,warp_mem_limit=256)
    valid = np.isfinite(dem)
    weights = gaussian_filter(valid.astype(float),sigma=2)
    smooth = gaussian_filter(np.where(valid,dem,0),sigma=2)/np.maximum(weights,1e-10)
    smooth[(weights<.999)|~valid] = np.nan
    return smooth,transform


def build(sites, dem, transform, relief=300., width=3000., peak=500., radius=80000.):
    from rasterio.warp import transform as project
    x,y = project('EPSG:4326','EPSG:5179',sites.lon.tolist(),sites.lat.tolist())
    xy = np.column_stack([x,y])
    blocked = np.zeros((len(sites),len(sites)),dtype=bool)
    rows = []
    for i in range(len(sites)):
        for j in range(i+1,len(sites)):
            length = np.linalg.norm(xy[i]-xy[j])
            if length>radius or length==0:
                continue
            n = max(2,int(np.ceil(length/1000))+1)
            line = np.linspace(xy[i],xy[j],n)
            cols,rr = (~transform)*(line[:,0],line[:,1])
            z = map_coordinates(dem,[rr-.5,cols-.5],order=1,mode='constant',cval=np.nan,prefilter=False)
            barrier = broad_barrier(z,length/(n-1),relief,width,peak)
            if barrier:
                blocked[i,j] = blocked[j,i] = True
            rows.append({'site_a':sites.site_id.iloc[i], 'site_b':sites.site_id.iloc[j],
                         'distance_km':length/1000, 'barrier':barrier,
                         'relief_m':float(z.max()-max(z[0],z[-1])) if np.isfinite(z).all() else None})
    return xy,blocked,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sites',type=Path,default=ROOT/'data/preprocessed_v1/sites.parquet')
    p.add_argument('--dem',type=Path,default=ROOT.parent/'data/terrain/original_expanded.vrt')
    p.add_argument('--output-dir',type=Path,default=ROOT/'data/ridge_regions_v1')
    p.add_argument('--relief-m',type=float,default=300)
    p.add_argument('--width-m',type=float,default=3000)
    p.add_argument('--peak-m',type=float,default=500)
    args=p.parse_args()
    if min(args.relief_m,args.width_m,args.peak_m)<=0:
        raise ValueError('Positive ridge thresholds required')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with connect() as con:
        read_view(con,'sites',args.sites)
        sites=con.execute('SELECT site_id,lat,lon FROM sites ORDER BY site_id').df()
    print('Aggregating existing terrain to 1 km and smoothing at 2 km...',flush=True)
    source_hashes=terrain_hashes(args.dem)
    dem,transform=prepare_dem(args.dem)
    xy,blocked,pairs=build(sites,dem,transform,args.relief_m,args.width_m,args.peak_m)
    pd.DataFrame(pairs).to_csv(args.output_dir/'ridge_pairs.csv',index=False)
    manifest={'status':'complete','sites_sha256':digest(args.sites),'dem_vrt_sha256':digest(args.dem),
        'dem_sources_sha256':source_hashes,
        'definition':'Broad straight-line ridge proxy, NOT hydrological watershed delineation',
        'parameters':{'resolution_m':1000,'smoothing_sigma_m':2000,'relief_m':args.relief_m,
                      'width_m':args.width_m,'peak_m':args.peak_m,'nearby_radius_m':80000},
        'barrier_pairs':int(blocked.sum()//2),'unknown_pairs':sum(r['barrier'] is None for r in pairs),
        'unknown_policy':'Unobserved terrain paths have no inferred ridge; inspect separately', 'candidates':{}}
    for kind in ('distance','ridge'):
        for k in (2,3,4):
            labels,info=partition(xy,blocked if kind=='ridge' else np.zeros_like(blocked),k)
            name=f'{kind}_k{k}'
            manifest['candidates'][name]=info
            if labels is not None:
                output=sites.assign(cluster=labels)
                output.to_csv(args.output_dir/f'{name}.csv',index=False)
                info['counts']=output.groupby('cluster').size().to_dict()
                colors=['#2563eb','#e11d48','#16a34a','#9333ea']
                points=(xy-xy.min(axis=0))/max(np.ptp(xy,axis=0))*700+35
                svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 800">',
                     '<rect width="800" height="800" fill="white"/>',f'<text x="20" y="22">{name}: disclosed coordinates</text>']
                for i,j in np.argwhere(np.triu(blocked,1)):
                    svg.append(f'<path d="M {points[i,0]} {780-points[i,1]} L {points[j,0]} {780-points[j,1]}" stroke="#aaa" opacity=".2"/>')
                for i,row in output.iterrows():
                    svg.append(f'<circle cx="{points[i,0]}" cy="{780-points[i,1]}" r="4" fill="{colors[row.cluster]}"><title>{row.site_id}: cluster {row.cluster}</title></circle>')
                svg.append('</svg>')
                (args.output_dir/f'{name}.svg').write_text('\n'.join(svg),encoding='utf-8')
            print(name,info,flush=True)
    manifest['artifacts']={f.name:digest(f) for f in args.output_dir.iterdir() if f.is_file()}
    save_json(args.output_dir/'manifest.json',manifest)


if __name__=='__main__':
    main()
