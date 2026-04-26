# Setup & run

Short path from clone to the Streamlit app.

## Prerequisites

- **Python 3.10+**
- **Git** (for the CLIP dependency in `requirements.txt`)

## 1. Clone and enter the project

```bash
git clone <repository-url>
cd IITR-Project
```

## 2. Virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install redis openai pydantic
pip install -r requirements.txt
```

`redis` and `openai` are required by the app but not listed in `requirements.txt`; install them as above.

> If `pip install -r requirements.txt` fails on `git+https://github.com/openai/CLIP.git`, ensure Git is installed and try again from a normal terminal (not a restricted sandbox).

## 4. Environment variables

```bash
cp .env.example .env
```

Edit **`.env`**:

| Variable | Required? | Purpose |
|----------|-----------|---------|
| `OPENAI_API_KEY` | Optional | Smarter routing + chat; omit to use keyword-only routing |
| `OPENAI_MODEL` | Optional | Defaults to `gpt-4o-mini` if unset |
| `FALKORDB_HOST`, `FALKORDB_PORT`, `FALKORDB_USER`, `FALKORDB_PASS` | Optional | Orders/invoices in chat; app starts without DB but those features fail gracefully |

Add FalkorDB lines to `.env` only if you have a graph instance (see your team’s connection details).

## 5. Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually **http://localhost:8501**).

---

## Troubleshooting

- **Port in use:** `streamlit run app.py --server.port 8502`
- **Heavy first import:** `similarity_engine` loads PyTorch/FAISS once; the first request can be slow.