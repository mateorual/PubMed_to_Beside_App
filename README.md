# PubMed to Bedside

A retrieval-augmented clinical literature assistant that takes a patient case and clinical questions, searches PubMed automatically, and returns a structured, cited answer grounded exclusively in retrieved evidence.

**Author:** Mateo Ruiz Alvarez — MSc AI, FAU Erlangen-Nürnberg  
**Seminar:** Large Language Models in Medicine, Summer 2026

> **Disclaimer:** This tool is for research and educational purposes only. It is a literature-support aid and does not replace clinical judgment. Do not enter real patient identifiers.

---

## Demo

![Pipeline demo](demo/Demo_01_CopyPaste_Short.gif)

*Paste a patient case description and clinical questions directly into the sidebar, run the 5-stage pipeline, and receive a structured, citation-audited answer grounded in PubMed evidence — all in under a minute.*

---

## What it does

1. Parses your patient case and questions
2. Extracts a structured patient profile (staging, DOI, margins, nodes, etc.)
3. Formulates 1–3 optimised PubMed MeSH queries via GPT-4o
4. Retrieves abstracts via NCBI E-utilities and screens them for relevance
5. Fetches open-access full text (PMC) for the top synthesis papers
6. Generates a consultation-note answer with inline PMID citations, grounded strictly in the retrieved papers
7. Audits every citation — hallucinated PMIDs are flagged prominently

---

## Pipeline architecture

```
Patient case + questions
        │
        ▼
┌─────────────────────────────┐
│  Stage 1 · Query formulator │  GPT-4o → PubMed MeSH queries + patient profile
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Stage 2 · PubMed fetcher   │  NCBI E-utilities → titles + abstracts (up to 3 query variants)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Stage 3 · Title screener   │  Lexical overlap pre-filter (no API cost)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Stage 4 · Abstract screener│  GPT-4o-mini scores each abstract 0–10 for relevance
└─────────────────────────────┘
        │  top-N passed papers
        ▼
┌─────────────────────────────┐
│  Stage 4b · Full-text fetch │  PMC Open Access full text for synthesis candidates only
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Stage 5 · Synthesizer      │  GPT-4o (or Llama 3.1 8B) → cited consultation-note answer
└─────────────────────────────┘
```

---

## Prerequisites

- Python 3.11+
- An **OpenAI API key** (required for query formulation, abstract screening, and synthesis)
- An **NCBI API key** (optional but recommended — increases PubMed rate limits; free at [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/))
- **Ollama** (optional — enables local open-source synthesis with Llama 3.1 8B)

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/mateorual/PubMed_to_Beside_App.git
cd PubMed_to_Beside_App

# 2. Create virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (and optionally NCBI_API_KEY, NCBI_EMAIL)

# 5. Run
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`.

---

## Using the app

### Input methods

| Method | How |
|--------|-----|
| **Paste text** | Type the case in the sidebar text area; enter questions one per line |
| **Upload `.txt`** | Plain text case; enter questions separately |
| **Upload `.json`** | JSON with `patient_description` and `questions` keys (see schema below) |
| **Select parsed case** | Choose a pre-parsed case from `data/parsed_cases/` |
| **Demo** | Click "Load demo case" for a built-in example |

JSON schema for upload:
```json
{
  "patient_description": "A 58-year-old man with ...",
  "questions": [
    "Would you recommend adjuvant radiation therapy?",
    "What dose and fractionation?"
  ]
}
```

Ready-to-use example files are provided in [`Live_Demonstration_Example/Dataset/`](Live_Demonstration_Example/Dataset/). For instance, upload [`Case_12.json`](Live_Demonstration_Example/Dataset/Case_12.json) directly via the "Upload file" option to run a tongue SCC case out of the box — no typing required. `.txt` variants (case description only) are also included for each case.

### Settings (sidebar)

| Setting | Default | Description |
|---------|---------|-------------|
| Max PubMed results per query | 30 | Papers retrieved per query variant |
| Max papers in synthesis | 5 | Top-ranked papers sent to the LLM |
| Include full text | On | Fetches PMC full text for synthesis candidates |
| Synthesis model | GPT-4o | Switch to open-source (Ollama/Llama 3.1 8B) here |

### Output sections

1. **Generated queries** — PubMed queries and extracted clinical keywords
2. **Pipeline trace** — evidence funnel (retrieved → screened → synthesis → full text)
3. **Evidence table** — all screened papers with relevance scores and full-text availability
4. **Answer** — consultation-note prose with clickable `[PMID:XXXXXXXX]` citation links
5. **Citation audit** — each cited sentence traced to its source excerpt
6. **Source documents** — expandable cards per paper (abstract + full text used)
7. **Developer trace** — raw JSON for debugging

---

## Open-source model (Ollama)

To use Llama 3.1 8B locally for synthesis instead of GPT-4o:

```bash
# Install Ollama: https://ollama.com
ollama pull llama3.1:8b
ollama serve
```

Then select **Open Source (Ollama)** in the sidebar. Query formulation and abstract screening always use OpenAI (GPT-4o-mini) regardless of this setting — only the final synthesis step switches to the local model.

---

## CLI usage

```bash
python app/cli.py \
  --case "A 58-year-old man with pT3b prostate cancer, Gleason 9, positive margins..." \
  --questions "Would you recommend adjuvant RT?" "What duration of ADT?" \
  --max_results 30 \
  --output data/evaluation_results/result.json
```

Or load a pre-parsed case:

```bash
python app/cli.py \
  --case_file data/parsed_cases/case1.json \
  --output data/evaluation_results/case1_result.json
```

---

## Adding your own cases (PDF)

Place PDF files in `data/cases/` and parse them:

```bash
# Parse all PDFs in data/cases/
python utils/pdf_parser.py

# Parse a single PDF
python utils/pdf_parser.py --pdf data/cases/MyCase.pdf
```

Parsed JSON files are saved to `data/parsed_cases/`. If the automatic parser misses the case/question boundary, edit `data/parsed_cases/manual_overrides.json`.

---

## Running tests

```bash
pytest tests/ -v
```

All tests are deterministic and do not require API keys.

---

## Project structure

```
PubMed_to_Beside_App/
├── app/
│   ├── streamlit_app.py     # Streamlit web UI
│   └── cli.py               # Command-line interface
├── pipeline/
│   ├── query_formulator.py  # Stage 1: PubMed query generation + patient profile
│   ├── pubmed_fetcher.py    # Stage 2: NCBI E-utilities retrieval
│   ├── title_screener.py    # Stage 3: lexical title pre-filter
│   ├── abstract_screener.py # Stage 4: LLM relevance scoring
│   ├── fulltext_fetcher.py  # Stage 4b: PMC full-text fetch
│   └── synthesizer.py       # Stage 5: cited answer generation + hallucination audit
├── utils/
│   ├── pdf_parser.py        # Parse Gray Zone case PDFs to JSON
│   └── evaluator.py         # ROUGE-L / BERTScore evaluation
├── tests/                   # Pytest test suite
├── data/
│   ├── cache/               # PubMed result cache (git-ignored)
│   ├── cases/               # Input PDF files (git-ignored)
│   ├── parsed_cases/        # Parsed case JSON files
│   └── evaluation_results/  # CLI output (git-ignored)
├── img/                     # Logos used in the Streamlit UI
├── Live_Demonstration_Example/
│   ├── How_to_Start_App.txt             # Quick-start instructions
│   ├── PatientDescriptions_and_Questions.docx  # Source case document
│   └── Dataset/
│       ├── Case_1.json / Case_1.txt     # Anal SCC with sacral metastasis
│       ├── Case_9.json / Case_9.txt     # High-risk prostate cancer post-prostatectomy
│       ├── Case_12.json / Case_12.txt   # Tongue SCC post-glossectomy
│       └── Test_Information/            # Expected output markdowns + clinical images
├── config.py                # All prompts, constants, and LLM client setup
├── requirements.txt
└── .env.example             # API key template
```

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `NCBI_API_KEY` | No | NCBI key for higher PubMed rate limits |
| `NCBI_EMAIL` | Recommended | Required by NCBI policy |
| `OLLAMA_BASE_URL` | No | Default: `http://localhost:11434/v1` |
| `OLLAMA_MODEL` | No | Default: `llama3.1:8b` |

---

## Safety and limitations

- All answers are grounded **exclusively** in retrieved PubMed papers — the model is instructed not to use training-data knowledge
- Hallucinated PMIDs (cited but not in the fetched set) are automatically detected and flagged
- Author names are taken directly from paper metadata — the model is forbidden from guessing
- Full-text retrieval is limited to **PMC Open Access** papers; many papers are abstract-only
- This is a research prototype, not a clinical decision support system
