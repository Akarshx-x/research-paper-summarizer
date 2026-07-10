# ResearchLens — Self-Supervised PyTorch Classifier · Dense-RAG Paper Chat

A hybrid **unsupervised extractive-abstractive** research paper summarizer,
extended with a **self-supervised PyTorch classifier** for a genuine
train/validation loss curve + accuracy metric, and a **Dense-RAG "Chat with
PDF"** feature for asking direct questions about the uploaded paper.

The summarizer itself is unchanged and still the primary engine: sentences
are selected directly from the source PDF via SciBERT embeddings + K-Means
clustering — no text is generated from scratch — with an optional Gemini
"polish" layer that smooths extracted sentences into fluent prose without
adding outside facts. "RAG" below refers specifically to the chat feature,
which does real query-driven retrieval-then-generation; the summarizer is
accurately described as unsupervised extractive clustering, not RAG.

```
PDF ──► Raw Text ──► Cleaned Sentences ──► SciBERT Embeddings ──┬─► K-Means Clusters ──► Summary (+ optional Gemini polish)
    (pdfplumber)   (regex cascade)      (768-dim, L2-norm)      │
                                                                 ├─► PyTorch MLP (SalienceClassifierMLP) ──► Accuracy + Loss Curves
                                                                 │
                                                                 └─► Chat query ──► NumPy cosine top-3 ──► Gemini (RAG answer)
```

---

## 1. Prerequisites

- **Python 3.10–3.12** (SciBERT/torch wheels are not yet published for every newer version)
- **Git**
- ~2 GB free disk space (PyTorch + SciBERT model download on first run)

Check your Python version:

```bash
python3 --version   # macOS/Linux
python --version    # Windows
```

---

## 2. Clone the repository

```bash
git clone <this-repo-url>
cd research-paper-summarizer
```

---

## 3. Create and activate a virtual environment

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

> If PowerShell blocks the activation script, run this once as admin:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Windows (Command Prompt)

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

You'll know it worked when your terminal prompt is prefixed with `(venv)`.

---

## 4. Install dependencies

Same command on every platform, once the venv is active:

```bash
pip install -r requirements.txt
```

Installs Streamlit, PyTorch (CPU build), Transformers, scikit-learn, pandas,
requests, and the rest of the pipeline. First install takes a few minutes —
PyTorch is the largest download. Gemini calls go over plain HTTPS via
`requests` — there is no `google-generativeai` SDK dependency.

---

## 5. (Optional) Enable Gemini — abstractive polish + RAG chat answers

The app works fully offline without this: the extractive summarizer, the
PyTorch classifier panel, and RAG retrieval (matched sentences) all still
work. Only the LLM-polished prose and the chat feature's generated answers
require a key — everything else has an offline fallback.

1. Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Copy the example secrets file and fill in your key:

   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

3. Open `.streamlit/secrets.toml` and replace the placeholder:

   ```toml
   GEMINI_API_KEY = "AIza..."
   ```

`secrets.toml` is gitignored — it will never be committed or pushed. Each
teammate needs their own key; **do not share keys over Slack/chat.**

**Cloud hosting:** paste the key into your host's secrets dashboard (e.g.
Streamlit Community Cloud → *Settings → Secrets*) — no file needed there.

**Free-tier quota:** Google's free `gemini-2.5-flash` tier caps out at
**20 requests/day per project** — easy to exhaust during active testing (it
resets daily). When exhausted, the app falls back to extractive output /
raw retrieved sentences automatically rather than failing, but polish and
chat answers stop working until the quota resets. Budget your testing
accordingly before a live demo.

---

## 6. Run the app

```bash
streamlit run app.py
```

Opens the app in your browser at `http://localhost:8501`. Drag and drop a
research paper PDF (a sample `paper.pdf` is included in this repo) and pick
a summary length in the sidebar. The dashboard renders, top to bottom:

- **PyTorch Supervised Validation** (full width) — real accuracy + train/
  validation loss curves from a classifier trained fresh on this document
  (skipped automatically on very short documents — see Troubleshooting)
- **Summary** (left column) — document analytics, then the extractive
  summary, optionally Gemini-polished, with a download button
- **RAG: Chat with PDF** (right column, side by side with Summary) — ask a
  question in the local input box, get an answer grounded in the top-3
  retrieved sentences from the paper
- **🔬 Advanced Academic Diagnostics** (collapsed expander at the bottom) —
  silhouette score + ROUGE-1/2/L against the paper's detected Abstract;
  still fully computed every run, just not surfaced by default

First run downloads the SciBERT model (~440 MB) — subsequent runs are fast
since it's cached locally by `transformers`.

---

## 7. (Optional) Run individual modules standalone

Useful for debugging or understanding each stage in isolation, no Streamlit
needed:

```bash
python paper_preprocessor.py      # Stage 1+2: PDF → cleaned sentences (synthetic demo)
python update_2_embeddings.py     # Stage 3: sentence embeddings (SciBERT)
python update_3_clustering.py     # Stage 4: K-Means clustering + summary (synthetic demo)
python supervised_classifier.py   # Stage 5: PyTorch MLP training smoke test (synthetic embeddings)
```

---

## Project Structure

```
research-paper-summarizer/
├── app.py                     ← Streamlit web app (entry point)
├── paper_preprocessor.py      ← PDF ingestion + text cleaning
├── update_2_embeddings.py     ← SciBERT sentence embeddings
├── update_3_clustering.py     ← K-Means clustering + summary generation
├── supervised_classifier.py   ← PyTorch MLP: weak-label generation + training
├── paper.pdf                  ← Sample test PDF
├── requirements.txt           ← Pinned Python dependencies
└── .streamlit/
    └── secrets.toml.example   ← Template for your Gemini API key
```

---

## Notes on the PyTorch classifier's accuracy metric

The classifier's training labels are **weak/self-supervised** — cosine
similarity of each sentence embedding to the document's own centroid,
top-vs-bottom **tertile** split into Salient/Background, with the ambiguous
middle tertile excluded from training and validation entirely — not
human-annotated ground truth, because no labeled salience corpus exists for
an arbitrary uploaded PDF. The reported accuracy and loss curves are
genuine measurements from a real 80/20 train/validation split on that proxy
label; nothing is hardcoded or clamped to a target range. Because the label
is derived from the same embeddings the classifier sees, real accuracy
varies by document — this is expected and is disclosed directly in the
app's supervised panel (including how many sentences were excluded as
ambiguous), not hidden.

## Notes on RAG retrieval quality

Chat retrieval uses **mean-pooled** SciBERT embeddings, not the [CLS]-token
embeddings used for K-Means clustering — raw [CLS] vectors are well-known
to retrieve poorly for short natural-language questions against long-form
technical prose (this is precisely the gap Sentence-BERT-style contrastive
fine-tuning was created to close). Mean-pooling measurably improved match
relevance with no new dependency. `paper_preprocessor.py` also filters out
whole-string-reversed text runs, a PDF-encoding defect observed directly in
one test document — this is a source-PDF text-layer issue, not an
extraction bug, and only that specific corruption pattern is caught; a PDF
with other encoding defects may still produce degraded retrieval.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` after install | Make sure your venv is activated — prompt should show `(venv)` |
| `pip install` fails on torch | Confirm Python is 3.10–3.12; older/newer versions may lack a prebuilt wheel |
| App says "LLM Polish — Skipped (Offline Mode)" | Expected if you haven't set up `secrets.toml` — the app still works, just extractive-only |
| App says "Supervised Validation — Skipped (Short Doc)" | Document has fewer than 30 sentences — too few to survive tertile-filtering and still leave a meaningful train/validation split |
| Gemini polish or chat answer fails/times out | REST calls enforce a strict 8s timeout; on failure the app falls back to extractive text or raw retrieved sentences rather than hanging. A `429` error specifically means the free-tier daily quota (20 requests/day) is exhausted, not a code error |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| Slow first run | Normal — SciBERT (~440 MB) downloads once and is cached afterward |
