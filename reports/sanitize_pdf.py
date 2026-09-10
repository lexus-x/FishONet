#!/usr/bin/env python3
"""Rebuild the FishONet report as a passive, attachment-safe PDF."""

from pathlib import Path

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "FishONet_Technical_Report_paper.pdf"
OUTPUT = ROOT / "FishONet_Technical_Report_Email.pdf"


def main() -> None:
    reader = PdfReader(SOURCE)
    writer = PdfWriter()

    for source_page in reader.pages:
        # Email attachments do not need interactive actions. Removing annotations
        # also eliminates URI/action dictionaries that aggressive scanners may flag.
        writer.add_page(source_page, excluded_keys=("/Annots", "/AA"))

    writer.add_metadata(
        {
            "/Title": "FishONet Technical Report",
            "/Author": "CWNU AIX",
            "/Subject": "Competition verification and reproducibility report",
            "/Creator": "CWNU AIX",
            "/Producer": "pypdf",
        }
    )
    writer.compress_identical_objects(remove_identicals=True, remove_orphans=True)

    with OUTPUT.open("wb") as output_file:
        writer.write(output_file)

    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes, {len(reader.pages)} pages)")


if __name__ == "__main__":
    main()
