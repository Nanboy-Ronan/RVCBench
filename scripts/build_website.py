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

FULL_TEXT_GUIDES = {
    'README': 'README.md',
    'quickstart': 'getting-started/quickstart.md',
    'installation': 'getting-started/installation.md',
    'metrics': 'evaluation/metrics.md',
    'adding_a_model': 'evaluation/adding_a_model.md',
    'core_suite': 'evaluation/core_suite.md',
    'datasets': 'evaluation/datasets.md',
    'api': 'reference/api.md',
    'cli': 'reference/cli.md',
    'faq': 'evaluation/faq.md',
    'versions': 'reproduction/versions.md',
}


def text_link(match, page_url, source_path):
    href = match[1]
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc or href.startswith('#'):
        return '](' + urljoin(page_url, href) + ')'
    public_names = {Path(path).name for path in FULL_TEXT_GUIDES.values()}
    public_names.update(path.name for directory in ('datasets', 'evaluation', 'getting-started',
                                                     'integrations', 'reference', 'reproduction')
                        for path in (ROOT / 'docs' / directory).glob('*.md'))
    is_docs_readme = parsed.path == 'README.md'
    if parsed.path.endswith('.md') and (Path(parsed.path).name in public_names) and (Path(parsed.path).name != 'README.md' or is_docs_readme):
        slug = Path(parsed.path).stem
        target = 'https://nanboy-ronan.github.io/RVCBench/docs/' + ('' if slug == 'README' else slug + '/')
        if parsed.fragment:
            target += '#' + parsed.fragment
    else:
        repository_path = Path('docs', source_path).parent.joinpath(parsed.path).resolve().relative_to(ROOT)
        target = 'https://github.com/Nanboy-Ronan/RVCBench/blob/main/' + repository_path.as_posix()
        if parsed.fragment:
            target += '#' + parsed.fragment
    return '](' + target + ')'


def main():
    output = ROOT / 'site'
    if output.exists():
        shutil.rmtree(output)
    subprocess.run([sys.executable, '-m', 'mkdocs', 'build', '--strict'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, 'website/build.py', '--output', str(output)], cwd=ROOT, check=True)
    shutil.copy2(ROOT / 'website/robots.txt', output / 'robots.txt')
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
    text = ['# RVCBench documentation', '', 'A plain-text export of the public English guides.', '']
    for guide, relative_path in FULL_TEXT_GUIDES.items():
        source = (ROOT / 'docs' / relative_path).read_text()
        source = re.sub(r'\A---\n.*?\n---\n', '', source, count=1, flags=re.S)
        url = 'https://nanboy-ronan.github.io/RVCBench/docs/' + ('' if guide == 'README' else guide + '/')
        # Keep relative document links usable in a concatenated plain-text export.
        source = re.sub(r'\]\(([^)\s]+)\)', lambda m: text_link(m, url, relative_path), source)
        text.extend(['---', 'Source: ' + url, '', source.strip(), ''])
    (output / 'llms-full.txt').write_text('\n'.join(text))
    print(f'Built homepage and documentation in {output}')


if __name__ == '__main__':
    main()
