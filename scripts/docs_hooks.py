"""Keep repository file links usable in both Markdown and the documentation site."""
import posixpath
import re
import json


GROUPED_DOC_DIRS = {
    'datasets',
    'evaluation',
    'getting-started',
    'integrations',
    'reference',
    'reproduction',
}
def on_files(files, *, config):
    """Keep public document URLs flat while organizing their source files.

    GitHub readers see topic-based directories under ``docs/``. MkDocs still
    treats each Markdown file as if it lived at the documentation root, so
    existing URLs such as ``/docs/metrics/`` and existing relative links remain
    stable.
    """
    for file in files:
        parts = file.src_uri.split('/')
        if len(parts) != 2 or parts[0] not in GROUPED_DOC_DIRS or not file.is_documentation_page():
            continue
        stem = posixpath.splitext(parts[1])[0]
        file.dest_uri = stem + '/index.html'
    return files


def on_page_markdown(markdown, *, page, config, files):
    def repository_link(match):
        target = match.group(1)
        if '://' in target or target.startswith(('#', 'mailto:')):
            return match.group(0)
        path, _, _ = target.partition('#')
        source_uri = page.file.edit_uri or page.file.src_uri
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source_uri), path))

        # These files live outside the reader documentation. Keep code, notebooks
        # and website-maintenance links on GitHub; MkDocs handles guide links.
        if resolved.startswith('../'):
            repository_path = posixpath.normpath(posixpath.join('docs', resolved))
            return '](' + config.repo_url + '/blob/main/' + repository_path + ')'
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
