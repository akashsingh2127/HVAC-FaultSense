"""Official LBNL RTU Dataset Downloader.

Acquires the official Simulated RTU dataset and documentation directly
from the Lawrence Berkeley National Laboratory (LBNL) repository:
https://faultdetection.lbl.gov/dataset/rtu/
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

PDF_URL = "https://fdddata.lbl.gov/data/Simulated_LBNL_FDD_Data_Sets_RTU/LBNL_FDD_Data_Sets_RTU.pdf"
ZIP_URL = "https://fdddata.lbl.gov/data/Simulated_LBNL_FDD_Data_Sets_RTU/LBNL_FDD_Data_Sets_RTU.zip"


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download a file with streaming chunks and progress display."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"File already exists: {dest_path} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
        return

    print(f"Downloading from {url} to {dest_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (HVAC-FaultSense-Downloader)"})
    with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rProgress: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="", flush=True)
            else:
                print(f"\rDownloaded: {downloaded / (1024*1024):.1f} MB", end="", flush=True)
        print()
    print(f"Download complete: {dest_path}")


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract a ZIP archive into the destination directory."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    print("Extraction complete.")


def acquire_lbnl_rtu_dataset(raw_dir: Path) -> dict:
    """Download the simulated RTU zip and documentation into data/raw/."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = raw_dir / "LBNL_FDD_Data_Sets_RTU.pdf"
    zip_path = raw_dir / "LBNL_FDD_Data_Sets_RTU.zip"
    simulated_dir = raw_dir / "simulated"

    download_file(PDF_URL, pdf_path)
    download_file(ZIP_URL, zip_path)

    if not any(simulated_dir.glob("*.csv")):
        extract_zip(zip_path, simulated_dir)

    # Count discovered files
    csv_files = list(simulated_dir.rglob("*.csv"))
    print(f"Discovered {len(csv_files)} CSV files in {simulated_dir}.")
    return {
        "pdf_path": str(pdf_path),
        "zip_path": str(zip_path),
        "simulated_dir": str(simulated_dir),
        "csv_count": len(csv_files),
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[3]
    raw_directory = base_dir / "data" / "raw"
    acquire_lbnl_rtu_dataset(raw_directory)

