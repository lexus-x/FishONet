#!/usr/bin/env python
"""Render report_design.html to a print-ready PDF with Chromium."""
import asyncio
import os
from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "report_design.html")
OUT = os.path.join(HERE, "FishONet_Technical_Report_design.pdf")

FOOT = """
<div style="width:100%; font-family:Helvetica,Arial,sans-serif; font-size:6.6pt;
            color:#A99C91; padding:0 16mm 0 20mm; display:flex;
            justify-content:space-between; letter-spacing:0.06em;">
  <span style="letter-spacing:.1em;">FISHONET &middot; TECHNICAL REPORT</span>
  <span style="font-variant-numeric:tabular-nums;">
    <span class="pageNumber"></span> / <span class="totalPages"></span>
  </span>
</div>"""

EMPTY = "<div></div>"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(f"file://{SRC}", wait_until="networkidle")
        await page.emulate_media(media="print")
        await page.pdf(
            path=OUT,
            format="A4",
            print_background=True,
            display_header_footer=False,
            header_template=EMPTY,
            footer_template=FOOT,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        )
        await browser.close()
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


asyncio.run(main())
