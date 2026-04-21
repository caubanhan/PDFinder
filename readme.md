# PDFinder

Local semantic search for PDFs from your terminal.

PDFinder helps you find relevant PDF files by meaning, not just by filename. It extracts text, creates embeddings with sentence-transformers, and uses FAISS for fast similarity search.

## Features

- Semantic search with natural language queries
- Fully local and offline (no cloud services)
- Fast retrieval with FAISS vector index
- Simple CLI workflow for Linux users

## Installation (Linux)

```bash
git clone https://github.com/<your-username>/PDFinder.git
cd PDFinder

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Index Your PDFs

Put your PDFs in a folder (default in this repo is pdf_test), then run:

```bash
python main.py
```

This creates:

- faiss_index.bin: FAISS vector index
- metadata_index.json: mapping from vector IDs to file/chunk metadata

## Create a pdfinder CLI Command

Create a small launcher script:

```bash
mkdir -p ~/bin
cat > ~/bin/pdfinder << 'EOF'
#!/bin/bash
/absolute/path/to/PDFinder/.venv/bin/python /absolute/path/to/PDFinder/finder.py "$@"
EOF

chmod +x ~/bin/pdfinder
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc   # or ~/.zshrc
source ~/.bashrc                                      # or source ~/.zshrc
```

Then search from anywhere:

```bash
pdfinder "create procedure sql"
```

## Usage Examples

```bash
pdfinder "database normalization examples"
pdfinder "operating systems deadlock prevention"
pdfinder "linear algebra eigenvalues notes"
```

## Notes

- No OCR support yet (image-only PDFs are not indexed well)
- First run may be slower because the embedding model is loaded/downloaded

## Tech Stack

- Python
- PyPDF2
- sentence-transformers (all-MiniLM-L6-v2)
- FAISS (faiss-cpu)

## License

MIT License.