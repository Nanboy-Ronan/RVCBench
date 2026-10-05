#!/usr/bin/env python3
"""Build the homepage and searchable documentation into one Pages artifact."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'site'
    if output.exists():
        shutil.rmtree(output)
    subprocess.run([sys.executable, 'docs/site-src/build.py'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-m', 'mkdocs', 'build', '--strict'], cwd=ROOT, check=True)
    for name in ('index.html', 'robots.txt', 'sitemap.xml', 'llms.txt'):
        shutil.copy2(ROOT / 'docs' / name, output / name)
    shutil.copytree(ROOT / 'docs/assets', output / 'assets')
    reports = ROOT / 'docs/validated_runs'
    if reports.is_dir():
        shutil.copytree(reports, output / 'validated_runs')
    (output / '.nojekyll').touch()
    print(f'Built homepage and documentation in {output}')


if __name__ == '__main__':
    main()
