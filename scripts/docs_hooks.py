"""Keep repository file links usable in both Markdown and the documentation site."""
import posixpath
import re
import json


def on_page_markdown(markdown, *, page, config, files):
    def repository_link(match):
        target = match.group(1)
        # These files live outside the reader documentation. Keep code, notebooks
        # and website-maintenance links on GitHub; MkDocs handles guide links.
        if target.startswith('../') or target.startswith('site-src/'):
            path = posixpath.normpath(posixpath.join('docs', posixpath.dirname(page.file.src_uri), target))
            return '](' + config.repo_url + '/blob/main/' + path + ')'
        return match.group(0)

    return re.sub(r'\]\(([^)\s]+)\)', repository_link, markdown)


def on_page_context(context, *, page, config, nav):
    base = 'https://nanboy-ronan.github.io/RVCBench/'
    schema = {
        '@context': 'https://schema.org',
        '@graph': [
            {
                '@type': 'TechArticle', '@id': page.canonical_url + '#article',
                'url': page.canonical_url, 'headline': page.title,
                'description': page.meta['description'], 'inLanguage': 'en',
                'isPartOf': {'@id': base + '#website'},
                'about': {'@id': base + '#software'},
            },
            {
                '@type': 'BreadcrumbList',
                'itemListElement': [
                    {'@type': 'ListItem', 'position': 1, 'name': 'RVCBench', 'item': base},
                    {'@type': 'ListItem', 'position': 2, 'name': 'Documentation', 'item': config.site_url},
                ] + ([] if page.url == '' else [
                    {'@type': 'ListItem', 'position': 3, 'name': page.title, 'item': page.canonical_url},
                ]),
            },
        ],
    }
    page.meta['seo_jsonld'] = json.dumps(schema, ensure_ascii=False).replace('<', '\\u003c')
    return context
