"""Download publicly available cloud-motion research; run from any directory."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
SOURCES = [
    ("01_miller_2018_satellite_model_coupling.pdf", "https://repository.library.noaa.gov/view/noaa/49543/noaa_49543_DS1.pdf"),
    ("02_carpentieri_2023_solarsteps_preprint.pdf", "https://arxiv.org/pdf/2305.07293v1"),
    ("03_straub_2024_helionet.pdf", "https://publica-rest.fraunhofer.de/server/api/core/bitstreams/93fbd665-f6ed-4edb-802b-ae2d0bf1e966/content"),
    ("04_papachristopoulou_2024_nextsense2.pdf", "https://amt.copernicus.org/articles/17/1851/2024/amt-17-1851-2024.pdf"),
    ("05_nmsc_amv_atbd_v1.pdf", "https://datasvc.nmsc.kma.go.kr/resources/common/pdf/NMSC-SCI-ATBD-AMV_v1.0.pdf"),
]


def download(source):
    name, url = source
    record = {"file": "pdf/" + name, "url": url}
    try:
        with urlopen(Request(url, headers={"User-Agent": "ResearchReferenceDownload/1.0"}), timeout=90) as response:
            data = response.read()
            record["final_url"] = response.url
        if not data.startswith(b"%PDF-"):
            raise ValueError("Response is not a PDF")
        (ROOT / record["file"]).write_bytes(data)
        record.update(status="ok", bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    except Exception as error:
        record.update(status="error", error=str(error))
    record["retrieved_utc"] = datetime.now(timezone.utc).isoformat()
    return record


if __name__ == "__main__":
    (ROOT / "pdf").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        records = list(pool.map(download, SOURCES))
    (ROOT / "downloads.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for record in records:
        print(record["status"], record["file"], record.get("bytes", record.get("error")))
    raise SystemExit(any(record["status"] != "ok" for record in records))
