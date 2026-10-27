"""Download public research PDFs and record integrity metadata (standard library)."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import urllib.request

ROOT = Path(__file__).resolve().parent
SOURCES = [
    ('01_gigoni_2018_day_ahead.pdf', 'https://arxiv.org/pdf/1903.06800'),
    ('02_holmgren_2018_pvlib.pdf', 'https://joss.theoj.org/papers/10.21105/joss.00884.pdf'),
    ('03_sharma_2021_probabilistic.pdf', 'https://arxiv.org/pdf/2101.08236'),
    ('04_lindberg_2023_trading.pdf', 'https://uu.diva-portal.org/smash/get/diva2%3A1770882/FULLTEXT01.pdf'),
    ('05_simeunovic_2021_spatiotemporal.pdf', 'https://arxiv.org/pdf/2107.13875v2'),
    ('06_nuno_2018_aggregated_errors.pdf', 'https://backend.orbit.dtu.dk/ws/files/149828757/revision_new_2_1_.pdf'),
    ('08_iea_2020_regional_forecasting.pdf', 'https://iea-pvps.org/wp-content/uploads/2020/07/IEA_PVPS_T16_regional_forecast.pdf'),
]

def download(item):
    name, url = item
    record = {'file': 'pdf/' + name, 'source_url': url,
              'checked_at_utc': datetime.now(timezone.utc).isoformat()}
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'SolarForecastResearch/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
            record['resolved_url'] = response.url
            record['content_type'] = response.headers.get('Content-Type')
        if not data.startswith(b'%PDF-'):
            raise ValueError('Response is not a PDF')
        target = ROOT / 'pdf' / name
        target.write_bytes(data)
        record.update(status='downloaded', bytes=len(data), sha256=sha256(data).hexdigest())
    except Exception as error:
        record.update(status='failed', error=str(error))
    return record

if __name__ == '__main__':
    (ROOT / 'pdf').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(download, SOURCES))
    (ROOT / 'downloads.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for record in records:
        print(record['file'], record['status'], record.get('bytes', record.get('error')))
