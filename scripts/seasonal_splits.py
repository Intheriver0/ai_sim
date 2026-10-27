"""Monthly outer evaluation with past-only, season-spanning inner selection."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

import pytz

from pipeline_common import connect, export, literal
from time_splits import attach_data, digest, interval, stamp, validate_inputs


def month_shift(day, offset):
    index = day.year * 12 + day.month - 1 + offset
    return day.replace(year=index // 12, month=index % 12 + 1, day=1)


def load_plan(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    zone = pytz.timezone('Asia/Seoul')
    start = zone.localize(datetime.strptime(cfg['evaluation_start_day'], '%Y-%m-%d'))
    end = zone.localize(datetime.strptime(cfg['evaluation_end_day_exclusive'], '%Y-%m-%d'))
    training_start = datetime.fromisoformat(cfg['training_start'])
    if (start.day != 1 or end != month_shift(start, 12) or cfg['inner_months'] != 12
            or cfg['generation_availability_lag_hours'] < 0 or training_start.tzinfo is None
            or cfg['inner_role_start_days'] != {'model_selection': 1, 'blend_selection': 11, 'composition_selection': 21}):
        raise ValueError('Expected twelve whole outer months and twelve historical inner months')

    def window(name, purpose, a, b):
        cutoff = a - timedelta(days=1) + timedelta(hours=17) - timedelta(hours=cfg['generation_availability_lag_hours'])
        if not training_start < cutoff < a < b:
            raise ValueError('Invalid training/evaluation chronology')
        stop_start, stop_end = a-timedelta(days=7), a-timedelta(days=1)
        fit_cutoff = stop_start-timedelta(days=1)+timedelta(hours=17)-timedelta(hours=cfg['generation_availability_lag_hours'])
        if not training_start < fit_cutoff < stop_start < stop_end <= cutoff:
            raise ValueError('Invalid early stopping chronology')
        return {'id': name, 'purpose': purpose, 'start': a, 'end': b,
                'training_start': training_start, 'training_cutoff': cutoff,
                'fit_cutoff': fit_cutoff,'early_stop_start': stop_start,'early_stop_end': stop_end}

    outers = []
    for i in range(12):
        a = month_shift(start, i)
        outer = window(a.strftime('outer_%Y_%m'), 'outer_evaluation', a, month_shift(a, 1))
        inner = []
        # Last complete target day whose actuals are available at the outer cutoff.
        available_end = outer['training_cutoff'].replace(hour=0, minute=0, second=0, microsecond=0)
        for j in range(-12, 0):
            m = month_shift(a, j)
            for role, first in cfg['inner_role_start_days'].items():
                left = m.replace(day=first)
                right = m.replace(day=first+10) if first < 21 else month_shift(m, 1)
                right = min(right, available_end)
                if right <= left:
                    raise ValueError('Availability lag removes an inner selection block')
                inner.append(window(f'{outer["id"]}_{m:%Y_%m}_{role}', role, left, right))
        outer['inner'] = inner
        outers.append(outer)
    return cfg, outers


def coverage(con, window):
    where = interval(window['start'], window['end'])
    hours, observed, complete = con.execute(f'''SELECT count(*), coalesce(sum(observed_sites),0),
        count(*) FILTER(WHERE observed_sites=253) FROM coverage WHERE {where}''').fetchone()
    expected = int((window['end']-window['start']).total_seconds()/3600)
    if hours != expected or observed == 0:
        raise ValueError(f'{window["id"]}: missing time axis or no observed targets')
    return {'hours': hours, 'observed_site_hours': observed,
            'missing_site_hours': hours*253-observed, 'complete_253_hours': complete,
            'settlement_available': complete > 0}


def window_views(con, window, allow_outer_evaluation=False):
    """Use after input validation. Observed-only diagnostics and full-group scoring are distinct."""
    if window['purpose'] == 'outer_evaluation' and not allow_outer_evaluation:
        raise ValueError('Outer evaluation is locked; freeze the selection procedure first')
    train = (f'ts >= {stamp(window["training_start"])} AND '
             f'ts <= {stamp(window["training_cutoff"])} AND training_eligible')
    con.execute(f'''CREATE OR REPLACE TEMP VIEW training_targets AS
        SELECT site_id,ts,target_kwh FROM targets WHERE {train}''')
    con.execute('''CREATE OR REPLACE TEMP VIEW training_features AS
        SELECT f.*,t.target_kwh FROM (SELECT * FROM features10 UNION ALL SELECT * FROM features17) f
        JOIN training_targets t USING(site_id,ts)''')
    for role, condition in [('fit', f'ts <= {stamp(window["fit_cutoff"])}'),
                            ('stop', interval(window['early_stop_start'],window['early_stop_end']))]:
        con.execute(f'CREATE OR REPLACE TEMP VIEW {role}_features AS SELECT * FROM training_features WHERE {condition}')
    where = interval(window['start'], window['end'])
    con.execute(f'''CREATE OR REPLACE TEMP VIEW diagnostic_features AS
        SELECT * FROM features10 WHERE {where} UNION ALL SELECT * FROM features17 WHERE {where}''')
    con.execute(f'''CREATE OR REPLACE TEMP VIEW diagnostic_actual AS
        SELECT site_id,ts,energy_raw_kwh AS energy FROM targets WHERE {where}''')
    con.execute(f'''CREATE OR REPLACE TEMP VIEW settlement_prices AS SELECT p.* FROM prices p
        JOIN (SELECT ts FROM targets WHERE {where} GROUP BY ts HAVING count(*)=253) h USING(ts)''')
    con.execute('''CREATE OR REPLACE TEMP VIEW settlement_actual AS
        SELECT a.* FROM diagnostic_actual a JOIN settlement_prices USING(ts)''')


def run(data_dir, config_path, output_dir):
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    cfg, outers = load_plan(config_path)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if data_dir in output_dir.parents:
        raise ValueError('Output must be outside preprocessing inputs')
    with connect() as con:
        attach_data(con, data_dir)
        validate_inputs(con)
        con.execute('''CREATE TEMP TABLE coverage AS SELECT p.ts,coalesce(t.n,0) AS observed_sites
            FROM prices p LEFT JOIN (SELECT ts,count(*) AS n FROM targets GROUP BY ts) t USING(ts)''')
        for outer in outers:
            outer['coverage'] = coverage(con, outer)
            outer['early_stop_coverage'] = coverage(con, {'id': outer['id'],
                'start': outer['early_stop_start'], 'end': outer['early_stop_end']})
            for inner in outer['inner']:
                inner['coverage'] = coverage(con, inner)
                if inner['end'] > outer['training_cutoff']:
                    raise ValueError('Inner labels cross the outer forecast origin')
            counts = con.execute(f'''SELECT s.site_id,count(t.ts) AS training_rows
                FROM sites s LEFT JOIN targets t ON s.site_id=t.site_id AND t.training_eligible
                AND t.ts >= {stamp(outer['training_start'])} AND t.ts <= {stamp(outer['training_cutoff'])}
                GROUP BY s.site_id ORDER BY s.site_id''').fetchall()
            outer['site_training_rows'] = dict(counts)
            outer['sites_without_training_history'] = sum(n == 0 for _, n in counts)
            outer['outer_scores_computed'] = False
        output_dir.mkdir(parents=True)
        export(con, ' UNION ALL '.join(f'''SELECT {literal(o['id'])} AS outer_id,ts,observed_sites,
            observed_sites=253 AS settlement_eligible FROM coverage WHERE {interval(o['start'],o['end'])}'''
            for o in outers), output_dir/'evaluation_calendar.parquet')
        manifest = {'status': 'complete', 'config': cfg, 'config_sha256': digest(config_path),
                    'input_sha256': {n: digest(data_dir/n) for n in ('report.json','sites.parquet',
                        'train_targets.parquet','train_prices.parquet','train_round10.parquet','train_round17.parquet')},
                    'outer_folds': outers, 'outer_evaluation_locked': True}
        (output_dir/'manifest.json').write_text(json.dumps(manifest, default=str, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/seasonal_splits_v2.json')
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    result = run(args.data_dir,args.config,args.output_dir)
    print(json.dumps([{'id': o['id'], **o['coverage']} for o in result['outer_folds']],indent=2))


if __name__ == '__main__':
    main()
