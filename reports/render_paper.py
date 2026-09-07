#!/usr/bin/env python
"""Render paper.html to the 3-page, two-column technical report PDF."""
import asyncio
import os
from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "paper.html")
OUT = os.path.join(HERE, "FishONet_Technical_Report_paper.pdf")


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(f"file://{SRC}", wait_until="networkidle")
        await page.emulate_media(media="print")
        # margins live in the CSS @page rule; passing them here would override it
        await page.pdf(path=OUT, format="A4", print_background=True,
                       display_header_footer=False,
                       margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                       prefer_css_page_size=True)
        await browser.close()
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


asyncio.run(main())
