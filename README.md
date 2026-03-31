# SWC-RAG

> Retrieval-Augmented Generation over PowerPoint presentations using **LangChain**, **Anthropic**, **ChromaDB**, and **sentence-transformers**.

SWC-RAG lets you ask questions about the slide decks stored in `data/` and get answers grounded in the text extracted from those presentations. It supports:

- a **terminal Q&A mode** for quick local usage
- a **single-question CLI mode** for scripts and testing
- a lightweight **Flask web app** with a JSON API

The app indexes `.pptx` files from `data/`, splits slide text into chunks, stores embeddings in **ChromaDB**, retrieves relevant chunks, and sends only that context to an Anthropic model for answering.

---

## Features

- **PowerPoint-first RAG** pipeline for `.pptx` files
- **Grounded answers** using retrieved slide context only
- **Persistent local vector store** with ChromaDB
- **Fast local embeddings** with `sentence-transformers/all-MiniLM-L6-v2`
- **Anthropic-powered responses** with configurable model selection
- **CLI + Web UI** in one project
- **Simple reindexing workflow** with a single flag

---

## How it works

1. **Load presentations** from `./data`
2. **Extract text** from each slide using `python-pptx`
3. **Chunk slide content** with `RecursiveCharacterTextSplitter`
4. **Embed chunks** using `all-MiniLM-L6-v2`
5. **Store or load vectors** from `./chroma_db`
6. **Retrieve top matches** for the user’s question
7. **Answer using only the retrieved context**

This makes the project ideal for slide-based Q&A, course material assistants, workshop decks, and internal knowledge retrieval from presentation files.

---

## Tech stack

- **Python**
- **LangChain**
- **Anthropic** (`langchain-anthropic`)
- **ChromaDB**
- **sentence-transformers**
- **python-pptx**
- **Flask**
- **python-dotenv**

---

## Project structure

```text
SWC-RAG/
├── data/                  # Put your .pptx files here
├── templates/
│   └── index.html         # Minimal web interface
├── main.py                # CLI, indexing, RAG chain, and Flask app
├── requirements.txt
├── .gitignore
└── Intro to LangChain.pdf
```

> `chroma_db/` is created at runtime and ignored by Git.

---

## Requirements

- Python **3.10+** recommended
- An **Anthropic API key**
- One or more `.pptx` files inside the `data/` directory

---

## Installation

```bash
git clone https://github.com/ahmedbahaj/SWC-RAG.git
cd SWC-RAG
python -m venv .venv
```

### Activate the virtual environment

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_api_key_here
ANTHROPIC_MODEL=claude-haiku-4-5
```

`ANTHROPIC_MODEL` is optional. If omitted, the app uses the default configured in `main.py`.

---

## Add your presentations

Place your PowerPoint files inside `data/`:

```text
data/
├── lecture1.pptx
├── workshop_intro.pptx
└── subfolder/
    └── deep_dive.pptx
```

Notes:

- The app scans `data/` **recursively**
- Temporary PowerPoint files such as `~$file.pptx` are ignored
- Only **extractable slide text** is indexed
- Images, scanned slides, and diagrams without text are **not OCR’d**

---

## Usage

### 1) Build the index and start interactive CLI mode

```bash
python main.py --reindex
```

Then ask questions in the terminal:

```text
Question: What is retrieval-augmented generation?
```

---

### 2) Ask one question directly

```bash
python main.py --question "What does slide 3 say about LangChain?"
```

Use `--reindex` together with it when your presentations changed:

```bash
python main.py --reindex --question "Summarize the main ideas in the deck."
```

---

### 3) Run the web app

```bash
python main.py --serve --reindex
```

By default, the app runs on:

```text
http://127.0.0.1:5000/
```

You can also customize host and port:

```bash
python main.py --serve --host 0.0.0.0 --port 8000
```

---

## API

### `POST /api/ask`

Ask a question through the JSON API.

#### Request

```json
{
  "question": "Summarize the main topic of the presentation."
}
```

#### Response

```json
{
  "answer": "..."
}
```

#### Error response

```json
{
  "error": "Missing or empty 'question' in JSON body."
}
```

### Example with `curl`

```bash
curl -X POST http://127.0.0.1:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the main idea of the slides?"}'
```

---

## CLI options

```bash
python main.py --help
```

Available options:

- `--reindex` → rebuild the Chroma index from `./data`
- `-q, --question` → ask one question and exit
- `--serve` → run the Flask web app instead of terminal mode
- `--host` → set the bind address for the web server
- `--port` → set the web server port

---

## Retrieval behavior

The current implementation is intentionally simple and easy to understand:

- documents are chunked with:
  - `chunk_size=1000`
  - `chunk_overlap=200`
- the retriever currently uses:
  - `k=6`
- answers are instructed to rely **only on the provided context**

That makes this repository especially useful as a clean educational example of a local RAG workflow over presentation files.

---

## Common issues

### `Set ANTHROPIC_API_KEY in your environment or .env file.`

Make sure your `.env` file exists and includes a valid key:

```env
ANTHROPIC_API_KEY=...
```

### `No text extracted from PowerPoint files`

Check that:

- your `.pptx` files are inside `data/`
- the slides contain actual text objects
- the files are not image-only or scanned slides

### Answers feel outdated after changing slides

Run again with:

```bash
python main.py --reindex
```

---

## Why this project is useful

SWC-RAG is a strong starter project if you want to learn or demonstrate:

- practical RAG fundamentals
- vector databases
- document chunking and retrieval
- grounded LLM prompting
- integrating LLM workflows into CLI and web apps

It is also a solid base for extending into:

- citations and source highlighting
- PDF support
- OCR for scanned slides
- conversation history
- multi-user web deployment
- authentication and document upload

---

## Suggested next improvements

If you plan to evolve this repo, these upgrades would add real value:

1. **Show cited slide references in the final answer**
2. **Display retrieved chunks in the UI**
3. **Support PDF and DOCX ingestion**
4. **Add OCR for image-based slides**
5. **Cache embeddings more explicitly and track index metadata**
6. **Add tests for ingestion and retrieval behavior**
7. **Dockerize the app for simpler deployment**

---

## Security note

Do not commit your `.env` file or API keys. The project already ignores `.env` and `chroma_db/`, which helps keep secrets and generated index data out of version control.

---

## Acknowledgments

Built with:

- [LangChain](https://www.langchain.com/)
- [Anthropic](https://www.anthropic.com/)
- [Chroma](https://www.trychroma.com/)
- [sentence-transformers](https://www.sbert.net/)
- [Flask](https://flask.palletsprojects.com/)

---

## License

No license is currently specified in this repository. If you want others to reuse or contribute confidently, adding a license file would be a good next step.
