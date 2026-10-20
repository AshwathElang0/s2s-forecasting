"""Build progress_report_standalone.html: progress_report.html with every local
image embedded as a data URI, so the single file can be emailed or opened
anywhere.

Usage:  python scripts/build_standalone_report.py
"""
import base64
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "progress_report.html"
OUT = ROOT / "progress_report_standalone.html"


def main():
    html = SRC.read_text(encoding="utf-8")
    cache = {}

    def data_uri(rel):
        if rel not in cache:
            cache[rel] = "data:image/png;base64," + base64.b64encode((ROOT / rel).read_bytes()).decode()
        return cache[rel]

    # <a href="x.png"><img src="x.png"></a>: embed the image once, drop the link
    # (a data-URI href would double the file size).
    html = re.sub(r'<a href="([^"]+\.png)"[^>]*>(<img[^>]*?)src="\1"([^>]*>)</a>',
                  lambda m: f'{m.group(2)}src="{data_uri(m.group(1))}"{m.group(3)}', html)
    html = re.sub(r'src="((?!data:|https?:)[^"]+\.png)"', lambda m: f'src="{data_uri(m.group(1))}"', html)
    OUT.write_text(html, encoding="utf-8")
    print(f"{len(cache)} images embedded -> {OUT.name} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
