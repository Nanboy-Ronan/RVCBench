#!/usr/bin/env python3
"""Build the homepage and searchable documentation into one Pages artifact."""
from pathlib import Path
import shutil
import subprocess
import sys
import re
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def text_link(match, page_url):
    href = match[1]
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc or href.startswith('#'):
        return '](' + urljoin(page_url, href) + ')'
    if parsed.path.endswith('.md') and '/' not in parsed.path:
        slug = Path(parsed.path).stem
        target = 'https://nanboy-ronan.github.io/RVCBench/docs/' + ('' if slug == 'README' else slug + '/')
        if parsed.fragment:
            target += '#' + parsed.fragment
    else:
        target = urljoin('https://github.com/Nanboy-Ronan/RVCBench/blob/main/docs/', href)
    return '](' + target + ')'


def main():
    output = ROOT / 'site'
    if output.exists():
        shutil.rmtree(output)
    subprocess.run([sys.executable, 'docs/site-src/build.py'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-m', 'mkdocs', 'build', '--strict'], cwd=ROOT, check=True)
    for name in ('index.html', 'robots.txt', 'llms.txt'):
        shutil.copy2(ROOT / 'docs' / name, output / name)
    shutil.copytree(ROOT / 'docs/assets', output / 'assets')
    reports = ROOT / 'docs/validated_runs'
    if reports.is_dir():
        shutil.copytree(reports, output / 'validated_runs')
    (output / '.nojekyll').touch()
    # Publish one complete sitemap. Omit build-time lastmod values: rebuilding
    # the site does not mean that every document's content changed.
    namespace = 'http://www.sitemaps.org/schemas/sitemap/0.9'
    ET.register_namespace('', namespace)
    docs_map = ET.parse(output / 'docs/sitemap.xml')
    for entry in docs_map.getroot():
        for date in list(entry.findall(f'{{{namespace}}}lastmod')):
            entry.remove(date)
    docs_map.write(output / 'docs/sitemap.xml', encoding='utf-8', xml_declaration=True)
    # Remove MkDocs' compressed copy containing the original timestamps.
    (output / 'docs/sitemap.xml.gz').unlink(missing_ok=True)
    root_map = ET.Element(f'{{{namespace}}}urlset')
    homepage = ET.SubElement(root_map, f'{{{namespace}}}url')
    ET.SubElement(homepage, f'{{{namespace}}}loc').text = 'https://nanboy-ronan.github.io/RVCBench/'
    root_map.extend(docs_map.getroot())
    ET.ElementTree(root_map).write(output / 'sitemap.xml', encoding='utf-8', xml_declaration=True)
    guides = ('README', 'quickstart', 'installation', 'metrics', 'adding_a_model', 'core_suite', 'datasets',
              'api', 'cli', 'faq', 'versions')
    text = ['# RVCBench documentation', '', 'A plain-text export of the public English guides.', '']
    for guide in guides:
        source = (ROOT / 'docs' / (guide + '.md')).read_text()
        source = re.sub(r'\A---\n.*?\n---\n', '', source, count=1, flags=re.S)
        url = 'https://nanboy-ronan.github.io/RVCBench/docs/' + ('' if guide == 'README' else guide + '/')
        # Keep relative document links usable in a concatenated plain-text export.
        source = re.sub(r'\]\(([^)\s]+)\)', lambda m: text_link(m, url), source)
        text.extend(['---', 'Source: ' + url, '', source.strip(), ''])
    (output / 'llms-full.txt').write_text('\n'.join(text))
    print(f'Built homepage and documentation in {output}')


if __name__ == '__main__':
    main()
