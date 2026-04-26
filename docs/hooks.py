import re

# all absolute media paths in .md doc files convert to relative mkdocs paths
def on_page_content(html, page, **kwargs):
    html = re.sub(r'src="docs/media/', 'src="media/', html)
    html = re.sub(r'href="docs/media/', 'href="media/', html)
    return html
