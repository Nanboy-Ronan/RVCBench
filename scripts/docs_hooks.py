"""Keep repository file links usable in both Markdown and the documentation site."""
import posixpath
import re


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
