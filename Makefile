DOCS_SRC := docs_src
SITE_DIR := docs

.PHONY: docs-sync-home docs-build docs-serve docs-clean

docs-sync-home:
	cp README.md $(DOCS_SRC)/index.md

docs-build: docs-sync-home
	mkdocs build --clean

docs-serve: docs-sync-home
	mkdocs serve

docs-clean:
	rm -rf $(SITE_DIR)
