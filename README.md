## Installation

This code has only been tested on Ubuntu/WSL. First install the following packages on linux:

```bash
$ sudo apt-get update
$ sudo apt-get install -y poppler-utils
```

Then create the and activate a conda environment using environment.yml:

```
$ conda env create -f environment.yml
$ conda activate zoterorag
```

Follow the steps for getting a [zotero API key](https://pyzotero.readthedocs.io/en/latest/) and add it to your .env file under the variable ```ZOTERO_API_KEY```.

## References

This repo builds on Nomic Embeds notebook for building a multimodal RAG system using their embedding model:

- [RAG Over PDFs with Nomic Embed Multimodal](https://docs.nomic.ai/atlas/embeddings-and-retrieval/guides/pdf-rag-with-nomic-embed-multimodal)