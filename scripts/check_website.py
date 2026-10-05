#!/usr/bin/env python3
"""Check built homepage/docs links, anchors and assets before Pages deployment."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import json
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1] / 'site'
SITE = urlsplit('https://nanboy-ronan.github.io/RVCBench/')


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.links = []
        self.ids = set()
        self.meta = {}
        self.canonical = None
        self.title = ''
        self.in_title = False
        self.schema_text = None
        self.schemas = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'meta':
            self.meta[attrs.get('name', attrs.get('property'))] = attrs.get('content', '')
        if tag == 'link' and attrs.get('rel') == 'canonical':
            self.canonical = attrs.get('href')
        if tag == 'title':
            self.in_title = True
        if tag == 'script' and attrs.get('type') == 'application/ld+json':
            self.schema_text = ''
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        for key in ('href', 'src'):
            if key in attrs:
                self.links.append(attrs[key])

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        if self.schema_text is not None:
            self.schema_text += data

    def handle_endtag(self, tag):
        if tag == 'title':
            self.in_title = False
        if tag == 'script' and self.schema_text is not None:
            self.schemas.append(json.loads(self.schema_text))
            self.schema_text = None


def main():
    pages = {path.resolve(): Page(path.read_text()) for path in ROOT.rglob('*.html')}
    if not pages or not (ROOT / 'docs/search/search_index.json').is_file():
        raise SystemExit('Build the website first; HTML and search index are required.')
    errors = []
    descriptions, titles, canonical_urls = set(), set(), set()
    for path, page in pages.items():
        if path.name != '404.html':
            relative = path.relative_to(ROOT).as_posix()
            expected = SITE.geturl() + relative.removesuffix('index.html')
            if page.canonical != expected:
                errors.append(f'{relative}: wrong canonical URL {page.canonical}')
            canonical_urls.add(expected)
            description = page.meta.get('description', '')
            if not description or description in descriptions or len(description) > 320:
                errors.append(f'{relative}: missing, duplicate or excessive description')
            descriptions.add(description)
            if not page.title or page.title in titles:
                errors.append(f'{relative}: missing or duplicate title')
            titles.add(page.title)
            for key in ('og:title', 'og:description', 'og:image', 'twitter:card'):
                if not page.meta.get(key):
                    errors.append(f'{relative}: missing {key}')
            if page.meta.get('og:url') != expected or 'noindex' in page.meta.get('robots', ''):
                errors.append(f'{relative}: incorrect indexing/social URL metadata')
            if not page.schemas:
                errors.append(f'{relative}: missing structured data')
        for link in page.links:
            url = urlsplit(link)
            if url.netloc and url.netloc != SITE.netloc:
                continue
            if url.scheme and url.scheme not in ('http', 'https'):
                continue
            if url.path.startswith('/'):
                if not url.path.startswith(SITE.path):
                    continue
                target = ROOT / unquote(url.path[len(SITE.path):])
            else:
                target = path.parent / unquote(url.path) if url.path else path
            target = target.resolve()
            if target.is_dir():
                target /= 'index.html'
            if not target.exists():
                errors.append(f'{path.relative_to(ROOT)}: missing {link}')
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                errors.append(f'{path.relative_to(ROOT)}: missing anchor {link}')
    sitemap = ET.parse(ROOT / 'sitemap.xml')
    locations = [node.text for node in sitemap.iter('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')]
    if set(locations) != canonical_urls or len(locations) != len(set(locations)):
        errors.append('Sitemap must contain every canonical content page exactly once')
    software = next(node for schema in pages[(ROOT / 'index.html').resolve()].schemas
                    for node in schema.get('@graph', []) if node.get('@id') == SITE.geturl() + '#software')
    version = re.search(r'^version = "([^"]+)"', (ROOT.parent / 'pyproject.toml').read_text(), re.M).group(1)
    if software.get('softwareVersion') != version or software.get('@type') != 'SoftwareApplication':
        errors.append('Homepage software entity/version does not match package metadata')
    if not (ROOT / 'llms-full.txt').is_file():
        errors.append('Missing full-text documentation export')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Checked {len(pages)} HTML pages: links, assets, metadata, JSON-LD and {len(locations)} sitemap URLs passed.')


if __name__ == '__main__':
    main()
