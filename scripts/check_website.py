#!/usr/bin/env python3
"""Check built homepage/docs links, anchors and assets before Pages deployment."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1] / 'site'
SITE = urlsplit('https://nanboy-ronan.github.io/RVCBench/')


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.links = []
        self.ids = set()
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        for key in ('href', 'src'):
            if key in attrs:
                self.links.append(attrs[key])


def main():
    pages = {path.resolve(): Page(path.read_text()) for path in ROOT.rglob('*.html')}
    if not pages or not (ROOT / 'docs/search/search_index.json').is_file():
        raise SystemExit('Build the website first; HTML and search index are required.')
    errors = []
    for path, page in pages.items():
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
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Checked {len(pages)} HTML pages: all local links, anchors and referenced assets exist.')


if __name__ == '__main__':
    main()
