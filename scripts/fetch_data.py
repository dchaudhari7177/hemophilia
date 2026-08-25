"""
Fetch the CDC hemophilia mutation datasets.

The CDC web server rejects non-browser clients, so a plain ``requests`` or
``curl`` call to the file URL returns 403 even though the files are public.
If that happens the script prints the two URLs and asks you to save them by
hand -- there is no automated bypass, and there should not be one.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
BASE = "https://www.cdc.gov/hemophilia/media/files"
LANDING = "https://www.cdc.gov/hemophilia/mutation-project/index.html"

FILES = {
    "CHAMP-Variant-List-2022.xlsx": "CHAMP: F8 variants, hemophilia A",
    "CHBMP-Variant-List-2022.xlsx": "CHBMP: F9 variants, hemophilia B",
}

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Referer": LANDING,
    "Accept": "*/*",
}


def fetch(name: str) -> bool:
    import urllib.error
    import urllib.request

    target = RAW / name
    if target.exists() and target.stat().st_size > 10_000:
        print(f"  have  {name} ({target.stat().st_size:,} bytes)")
        return True
    req = urllib.request.Request(f"{BASE}/{name}", headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        print(f"  FAIL  {name}: {exc}")
        return False
    RAW.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print(f"  saved {name} ({len(data):,} bytes)")
    return True


def export_champ_csv() -> None:
    """Write the flat CSV the pipeline reads by default."""
    import pandas as pd

    src = RAW / "CHAMP-Variant-List-2022.xlsx"
    dst = RAW / "champ.csv"
    if dst.exists() or not src.exists():
        return
    pd.read_excel(src, sheet_name="CHAMP Variant List", header=0).to_csv(
        dst, index=False)
    print(f"  wrote {dst.name}")


def main() -> int:
    print("Fetching CDC hemophilia mutation datasets...")
    ok = all(fetch(n) for n in FILES)
    if ok:
        export_champ_csv()
        print("Done.")
        return 0
    print("\nThe CDC server blocked the automated request (this is normal).")
    print(f"Open {LANDING} in a browser and save these two files into "
          f"{RAW}:")
    for name, what in FILES.items():
        print(f"  - {name}  ({what})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
