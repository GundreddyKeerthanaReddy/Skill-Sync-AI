"""Optional downloader for the public Mendeley career-path dataset."""
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
# Public dataset landing page and Mendeley public-file endpoint (version 1).
LANDING_PAGE = "https://data.mendeley.com/datasets/cns7h4b9g7/1"
DOWNLOAD_URL = "https://data.mendeley.com/public-files/datasets/cns7h4b9g7/files/Career_Path_Dataset_with_Career.csv?file=1"

def download_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    target = DATA_DIR / "career_path_public.csv"
    try:
        request = Request(DOWNLOAD_URL, headers={"User-Agent": "SkillSync-AI-educational-project/1.0"})
        with urlopen(request, timeout=25) as response:
            content = response.read()
        if len(content) < 1000 or not (content[:1000].lower().count(b",") > 1):
            raise ValueError("Downloaded content did not look like a CSV file.")
        target.write_bytes(content)
        return str(target)
    except (URLError, HTTPError, TimeoutError, ValueError, OSError) as exc:
        raise RuntimeError(f"Public dataset download failed: {exc}. The bundled fallback dataset remains available.") from exc

if __name__ == "__main__":
    print(download_dataset())
