# Semantic Query Routing System
### Privacy-Aware Two-Tier Edge–Fog Routing for Visually Impaired Smart Glasses Users

**NSF IRES | HPCC Lab UNT | IMDEA Networks**

---

## Overview

This system routes natural language queries from smart glasses users across two compute tiers — a private **edge device** (smartphone/laptop) and a more powerful **fog server** — while enforcing strict privacy guarantees. Personal data never leaves the edge device.

The system classifies every query into one of three categories:

| Class | Description | Route |
|---|---|---|
| **Personal** | About the user's own data (health, documents, bookings) | Edge only |
| **Environmental** | About physical surroundings (signs, navigation, nearby places) | Fog only |
| **Mixed** | Requires both personal lookup and environmental context | Both tiers |

### Example Routing Decisions

```
"What is my blood pressure medication?"
  → Personal → Edge (private LLM, never leaves device)

"Is there a pharmacy nearby?"
  → Environmental → Fog (vision-capable GPU server)

"What medication am I holding and is there a pharmacy nearby?"
  → Mixed → Edge (personal part) + Fog (environmental part) → Fused on Edge

"Where is my gate?"
  → Mixed (implicit) → Edge fetches gate number → Fog navigates to it
```

---

## System Architecture

```
Smart Glasses (camera + mic)
         │
         ▼
  ┌─────────────────────────────────────────────┐
  │            EDGE DEVICE (Smartphone)          │
  │                                             │
  │  ┌──────────────────────────────────────┐  │
  │  │        C1: Query Classifier          │  │
  │  │   5-rule pipeline + swappable ML     │  │
  │  │   backend (MiniLM-LR / SetFit /      │  │
  │  │   FastFit / Cosine)                  │  │
  │  └──────────┬───────────┬──────────────┘  │
  │             │           │                  │
  │         Personal      Mixed   Environmental │
  │             │           │            │      │
  │             │    ┌──────┴──────┐     │      │
  │             │    │ C2 Decompose│     │      │
  │             │    │ C3 Depend.  │     │      │
  │             │    └──────┬──────┘     │      │
  │             │    Parallel│Sequential  │      │
  │             │           │            │      │
  │  ┌──────────▼───────────▼────────┐   │      │
  │  │   C5: Edge Model (Llama 3.2:3B)   │      │
  │  │       Ollama on localhost     │   │      │
  │  └──────────────────────────────┘   │      │
  │             │                        │      │
  └─────────────│────────────────────────│──────┘
                │  [personal answers     │
                │   stay on edge]        │  [env queries only]
                │                        ▼
         ┌──────────────────────────────────────┐
         │         FOG SERVER (GPU)              │
         │  C5: Fog Model (Llama 3.2-Vision:11B) │
         │         Ollama on remote host         │
         └──────────────────────────────────────┘
                          │
                          ▼ [fog response returns]
                ┌─────────────────────┐
                │   C6: Response      │
                │   Fusion (on edge)  │
                │  Personal + Env →   │
                │   Single Answer     │
                └─────────────────────┘
```

**Privacy invariant:** Personal answers are always fused on the edge. The fog server only ever receives environmental sub-queries.

---

## Components

| ID | File | Description |
|----|------|-------------|
| C1 | `router/classifier.py` | 5-rule query classifier wrapping a swappable ML backend |
| C1-backends | `router/classifiers/` | Four interchangeable ML backends (see below) |
| C2 | `router/decomposer.py` | Splits Mixed queries into personal + environmental sub-queries |
| C3 | `router/dependency.py` | Decides parallel vs. sequential dispatch |
| C4 | `router/dispatch.py` | Async dispatcher — routes sub-queries, enforces hard privacy block |
| C5a | `edge/model.py` | `EdgeModelClient` — wraps Ollama on localhost (llama3.2:3b) |
| C5b | `edge/model.py` | `FogModelClient` — wraps Ollama on remote GPU server (llama3.2-vision:11b) |
| C6 | `edge/fusion.py` | Fuses edge + fog responses into a single answer (runs on edge) |
| — | `edge/session.py` | Rolling context buffer for pronoun resolution across turns |
| — | `edge/pipeline.py` | End-to-end orchestrator — ties C1–C6 together |
| — | `context_store/edge_store.py` | SQLite store for personal context (medications, bookings, contacts) |
| — | `context_store/fog_store.py` | FAISS semantic index for environmental context |
| — | `fog/server.py` | Fog server entry point |
| — | `config.py` | Loads `config.yaml` |
| — | `run_simulation.py` | Interactive/batch demo runner |
| — | `scripts/seed_edge_db.py` | Seeds the edge SQLite store with demo personal data |
| — | `scripts/evaluate.py` | Single-backend evaluation (precision/recall/F1/latency) |
| — | `scripts/compare_classifiers.py` | Trains all backends on same split and prints comparison table |

---

## C1 Classifier — 5-Rule Decision Pipeline

Every query passes through these rules in order. The first rule that fires wins.

```
Query
  │
  ▼
Rule 0: Implicit Mixed patterns
  e.g. "Where is my gate?" → needs personal lookup (gate #) then environmental
       navigation — Mixed even without a conjunction.
  Patterns: "where is my X", "how do I get to my X", "how far is my X",
            "is this the X for my Y?", etc.
  → If match → Mixed (conf 0.90)
  │
  ▼
Rule 0.5: Explicit Mixed (conjunction + personal + environmental signals)
  e.g. "My appointment is at 3pm and is this the right building?"
  Requires: conjunction ("and", "while", etc.)
          + personal signal (my / am I / I'm / do I)
          + environmental signal (nearby / this building / right terminal / etc.)
  Fires BEFORE the keyword backstop — so "my appointment ... this building"
  routes to Mixed instead of being force-Personal by the backstop.
  → If match → Mixed (conf 0.88)
  │
  ▼
Rule 1: Keyword backstop (hard privacy enforcement)
  Explicit privacy keywords from config.yaml force Personal regardless of
  ML output. Examples: "my medication", "my passport", "my insurance"
  → If keyword found → Personal (conf 1.00)
  │
  ▼
Rule 2+3: ML Backend (swappable — see Classifier Backends below)
  Calls backend.predict_proba(query) → {"Personal": p, "Environmental": e, "Mixed": m}
  If top_conf >= confidence_threshold (default 0.65) → use result directly
  │
  ▼
Rule 2 (low-confidence handler):
  top_label == Mixed       → trust it (Mixed gets lower confidence by nature;
                              the personal sub-query still stays on edge)
  top_label == Environmental
    + no personal signals  → trust Environmental (safe to send to fog)
    + personal signals     → safe-default to Personal
  top_label == Personal    → Personal (always safe)
```

### Why This Layered Design?

The ML model alone has weak class boundaries with small training sets, especially for Mixed. The rule layers compensate:

- **Rules 0 / 0.5** catch structural Mixed patterns the ML model sees rarely, with near-perfect precision
- **Rule 1** enforces hard privacy for sensitive keywords regardless of ML output
- **Rule 2** fixes the original bug where all low-confidence predictions defaulted to Personal, silently killing all Mixed routing

---

## Classifier Backends

The ML step (Rules 2+3) is a **swappable backend**. All backends implement the same interface (`BaseQueryClassifier`) and return the same `predict_proba()` dict. The rest of the pipeline is completely unchanged regardless of which backend runs.

| Backend | Algorithm | Train time | Inference | Install |
|---|---|---|---|---|
| `minilm_lr` **(default)** | Frozen MiniLM encoder + scikit-learn LogisticRegression | ~2–5 s | ~7 ms | included |
| `cosine` | Nearest-centroid cosine similarity baseline | ~1 s | ~5 ms | included |
| `setfit` | SetFit few-shot fine-tuning (contrastive pairs) | ~30 s–2 min | ~15 ms | `pip install setfit` |
| `fastfit` | FastFit (IBM) repeated-classification loss | ~10–60 s | ~12 ms | `pip install fastfit` |

### Switching Backends

**In `config.yaml`** (affects the live pipeline):
```yaml
classifier_backend: "setfit"   # minilm_lr | cosine | setfit | fastfit
```

**Via `evaluate.py`** (for testing only, doesn't change config):
```bash
python scripts/evaluate.py --backend setfit
```

### Saved Model Locations

Each backend saves its weights independently so they can all coexist:

```
models/classifier/
  minilm_lr/head.pkl       ← LogisticRegression head
  cosine/centroids.pkl     ← class centroid vectors
  setfit/                  ← full SetFit model directory
  fastfit/                 ← FastFit model directory
```

---

## C2 Decomposer — Mixed Query Splitting

Splits a Mixed query into exactly two sub-queries. Strategies tried in order:

1. **Implicit personal detection** — "Where is my gate?" → `personal="What is my gate?"`, `env="Where is my gate?"` (triggers sequential dispatch)
2. **Conjunction split** — split on "and", "but also", "as well as", etc., then assign personal/environmental sides by possessive presence
3. **spaCy dependency parse** — clause boundary detection via ROOT/conj verbs
4. **SLM fallback** — prompt the edge model with a JSON decomposition request
5. **Last resort** — treat entire query as Personal (safe default)

---

## C3 Dependency Detector — Parallel vs Sequential

Determines dispatch mode for Mixed queries:

| Mode | When | Example |
|------|------|---------|
| **Sequential** | Environmental sub-query contains unresolved pronoun | "Where is it?" where "it" = gate number from edge |
| **Sequential** | Query was detected as implicitly personal | "Where is my gate?" |
| **Sequential** | Explicit markers present ("based on that", "if so") | "If I have penicillin allergy, should I avoid this?" |
| **Parallel** | No dependency detected | "What medication am I on and is there a pharmacy nearby?" |

Sequential dispatch: edge runs first, its answer is injected into the fog prompt.

---

## C6 Response Fusion

Fuses edge and fog answers into a single response. Runs **on edge** to prevent personal data leakage.

Two modes (configurable):
- **Template fusion** (default, fast) — fills structured template based on route type
- **LLM fusion** — prompts the edge SLM to write a natural combined response

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd semantic_router
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Optional backends:**
```bash
pip install setfit     # for SetFit backend
pip install fastfit    # for FastFit backend
```

### 2. Configure

```bash
copy config.example.yaml config.yaml
```

Edit `config.yaml`:
```yaml
simulation_mode: true
fog_server_url: "http://YOUR_FOG_SERVER_IP:11435"
classifier_backend: "minilm_lr"   # minilm_lr | cosine | setfit | fastfit
```

### 3. Seed personal context store

```bash
python scripts/seed_edge_db.py
```

### 4. Run the demo

```bash
python run_simulation.py                   # interactive
python run_simulation.py --batch           # all demo queries
python run_simulation.py --query "..."     # single query
python run_simulation.py --production      # requires Ollama
```

### 5. Production setup (optional)

```bash
# Edge (laptop)
ollama pull llama3.2:3b && ollama serve

# Fog server (GPU machine)
ollama pull llama3.2-vision:11b
ollama serve --host 0.0.0.0
```

---

## Training the Classifier

Training happens automatically on first run when no saved model exists.

```bash
# Force retrain the active backend
del models\classifier\minilm_lr\head.pkl   # Windows
python run_simulation.py                   # retrains on startup
```

### Training Data Format

`dataset/training_data.json` — one entry per line:

```json
{"query": "What is my blood pressure medication?", "label": "Personal"}
{"query": "Is there a pharmacy nearby?",           "label": "Environmental"}
{"query": "What medication am I holding and is there a pharmacy nearby?", "label": "Mixed"}
```

Labels must be exactly `Personal`, `Environmental`, or `Mixed`.

### Training Tips

- Aim for **balanced classes** — roughly equal counts per class
- Mixed needs the most examples — it is the hardest class to learn
- Queries with "my" in a spatial context ("door to my left") are Environmental, not Personal — add them explicitly
- More data always helps; the rule layers (Rules 0/0.5) cover common patterns so the ML backend only needs to handle edge cases

---

## Evaluation

### Single backend

```bash
python scripts/evaluate.py                        # default: minilm_lr
python scripts/evaluate.py --backend cosine
python scripts/evaluate.py --backend setfit
python scripts/evaluate.py --backend fastfit
python scripts/evaluate.py --backend minilm_lr --test path/to/test.json
python scripts/evaluate.py --backend minilm_lr --retrain
```

Output: per-class precision/recall/F1, macro F1, confusion matrix, per-class accuracy, confidence distribution, top misclassified examples.

### Compare all backends

```bash
python scripts/compare_classifiers.py                          # all four
python scripts/compare_classifiers.py --backends cosine minilm_lr
python scripts/compare_classifiers.py --test-size 0.25 --seed 0
```

Example output:

```
================================================================
  CLASSIFIER COMPARISON
================================================================
  Backend        MacroF1   Env F1   Mix F1   Per F1  Lat(ms) Train(s) Size(MB) Errors
  ────────────────────────────────────────────────────────────────────────────────────
  cosine           0.712    0.680    0.648    0.808      2.1      1.0      0.0     28
  minilm_lr        0.912    0.940    0.908    0.887      7.3      4.0      0.1      5
  setfit           0.931    0.952    0.926    0.915     15.4     45.0    412.0      4
  fastfit          0.938    0.960    0.934    0.921     12.1     38.0    380.0      3
================================================================
```

### Research benchmark

Use this when choosing the classifier for the routing paper/system, because it
optimizes for privacy and edge constraints instead of only macro F1.

```bash
python scripts/research_benchmark.py
python scripts/research_benchmark.py --test-size 0.25 --seed 0
python scripts/research_benchmark.py --json-output research_results.json
python scripts/research_benchmark.py --keep-trained-models
```

Default backends: `minilm_lr`, `setfit`, `fastfit`.
By default, benchmark training restores existing `models/classifier/...` artifacts after each backend. Use `--keep-trained-models` only when you want the benchmark-trained models to replace the saved ones.

Reported metrics:

- **Personal recall** - how often true Personal queries stay Personal
- **Mixed-query F1** - quality on the hardest routing class
- **P->Fog error** - true Personal queries predicted as Mixed/Environmental, plus true Mixed queries predicted as Environmental
- **Direct P->Fog error** - true Personal or Mixed queries predicted as Environmental
- **Latency** - average and p95 classifier latency after warmup
- **Size/RSS** - saved model size and process memory delta when `psutil` is installed
- **Balance score** - weighted score: Personal recall 0.30, P->Fog safety 0.25, Mixed F1 0.25, latency 0.10, model size 0.10

### Current Results — `minilm_lr` backend

```
                precision  recall  f1-score  support
Environmental     1.000    0.884     0.938      86
Mixed             0.961    0.860     0.908      86
Personal          0.802    0.977     0.881      87

accuracy                             0.907     259
macro avg         0.921    0.907     0.909     259

Macro F1: 0.909
Avg confidence (correct)  : 0.860
Avg confidence (incorrect): 0.758
```

---

## Privacy Model

```
┌──────────────────────────────────────────────────────┐
│  HARD PRIVACY RULES (enforced in code, not config)   │
│                                                      │
│  1. Personal sub-query answers NEVER sent to fog     │
│     Enforced in: router/dispatch.py                  │
│                                                      │
│  2. Keyword backstop in classifier.py                │
│     Privacy keywords → Personal regardless of ML    │
│                                                      │
│  3. Response fusion runs on edge                     │
│     Fog only receives environmental sub-queries      │
│     Enforced in: edge/fusion.py                      │
│                                                      │
│  4. Sequential dispatch injects only the             │
│     necessary personal fact (e.g., gate number),     │
│     not raw personal data                            │
│     Enforced in: router/dispatch.py                  │
└──────────────────────────────────────────────────────┘
```

### Privacy Keywords (`config.yaml`)

Queries containing these phrases are always routed to edge regardless of ML output:

```
my medication, my prescription, my blood pressure, my health,
my doctor, my appointment, my passport, my id, my wallet,
my bank, my insurance, my diagnosis, my contact, my booking,
my flight, my calendar, my schedule, my address, my phone number
```

---

## Session Context

`SessionManager` maintains a rolling buffer of the last N query–response pairs (default: 10). Before routing each query it checks for unresolvable pronouns ("Where is it?", "What time does it close?"). If found, it injects the last 2 session turns as context.

**Image exception:** if a camera frame is attached, the image resolves "this/it/that" — no injection needed.

---

## Project Structure

```
semantic_router/
├── config.yaml                     # Runtime configuration
├── config.example.yaml             # Template
├── config.py                       # Config loader
├── run_simulation.py               # Demo runner
│
├── router/
│   ├── classifier.py               # C1: 5-rule pipeline (backend-agnostic)
│   ├── classifiers/
│   │   ├── base.py                 # Abstract base: predict_proba() interface
│   │   ├── minilm_lr.py            # Frozen MiniLM + LogisticRegression (default)
│   │   ├── cosine_clf.py           # Nearest-centroid cosine baseline
│   │   ├── setfit_clf.py           # SetFit few-shot (pip install setfit)
│   │   ├── fastfit_clf.py          # FastFit IBM (pip install fastfit)
│   │   └── __init__.py             # Factory: get_classifier("setfit")
│   ├── decomposer.py               # C2: Mixed query splitter
│   ├── dependency.py               # C3: Parallel vs sequential detector
│   └── dispatch.py                 # C4: Async dispatcher + privacy enforcement
│
├── edge/
│   ├── pipeline.py                 # End-to-end orchestrator
│   ├── model.py                    # C5: Edge + Fog Ollama clients
│   ├── fusion.py                   # C6: Response fuser (runs on edge)
│   └── session.py                  # Rolling context buffer
│
├── context_store/
│   ├── edge_store.py               # SQLite personal context
│   └── fog_store.py                # FAISS environmental context index
│
├── fog/
│   └── server.py                   # Fog server entry point
│
├── dataset/
│   └── training_data.json          # Labelled queries for training
│
├── models/
│   └── classifier/
│       ├── minilm_lr/head.pkl      # LR head (auto-generated)
│       ├── cosine/centroids.pkl    # Cosine centroids (auto-generated)
│       ├── setfit/                 # SetFit model directory (auto-generated)
│       └── fastfit/                # FastFit model directory (auto-generated)
│
├── data/
│   ├── edge_context.db             # SQLite personal data (auto-generated)
│   ├── fog_index.faiss             # FAISS index (auto-generated)
│   └── fog_metadata.json
│
├── scripts/
│   ├── seed_edge_db.py             # Populates edge_context.db with demo data
│   ├── evaluate.py                 # Single-backend evaluation (--backend flag)
│   ├── compare_classifiers.py      # Side-by-side comparison of all backends
│   ├── research_benchmark.py       # Privacy/edge-focused research benchmark
│   ├── balance_dataset.py          # Balances class counts in training_data.json
│   └── convert_vizwiz.py           # Converts VizWiz dataset to training format
│
└── tests/
    └── test_pipeline.py            # Integration tests
```

---

## Configuration Reference

```yaml
# config.yaml

simulation_mode: true         # true = mock LLMs (no Ollama needed)
log_level: INFO

# Fog server
fog_server_url: "http://localhost:11435"
fog_model: "llama3.2-vision:11b"
fog_timeout_sec: 30

# Edge model
edge_ollama_url: "http://localhost:11434"
edge_model: "llama3.2:3b"
edge_timeout_sec: 20

# Classifier
classifier_model: "all-MiniLM-L6-v2"   # SentenceTransformer encoder name
classifier_backend: "minilm_lr"         # minilm_lr | cosine | setfit | fastfit
classifier_confidence_threshold: 0.65   # Below this → low-confidence handler
use_keyword_backstop: true              # Force Personal on privacy keywords

# Context stores
edge_db_path: "data/edge_context.db"
fog_faiss_path: "data/fog_index.faiss"
fog_metadata_path: "data/fog_metadata.json"

# Session
session_buffer_size: 10                 # Rolling window of last N turns
session_inject_on_unresolvable: true

# Privacy keywords — force Personal regardless of classifier output
privacy_keywords:
  - "my medication"
  - "my prescription"
  - "my passport"
  # ... (see config.yaml for full list)
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Known Limitations

- **Keyword backstop over-fires on some Mixed queries** — queries like "Can I take my medication here?" get forced Personal because the backstop doesn't see the environmental component. Rule 0.5 catches most of these (requires conjunction or structural pattern), but pure-possessive Mixed queries without conjunctions still fall through.
- **"am I in [location]?" ambiguity** — "What section am I in?" is Environmental but "am I" is a personal signal. The low-confidence handler sends it to Personal. Adding such examples to the training data resolves this.
- **SetFit / FastFit require optional installs** — `pip install setfit` or `pip install fastfit`. If not installed, `compare_classifiers.py` skips them gracefully with a message.
- **Simulation mode mock responses are hardcoded** — production requires real Ollama models on both edge and fog.
