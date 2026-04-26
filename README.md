# VARYNT — AI Engineer Assessment

A 24-hour technical challenge for the AI Architect + ML Engineer role at DRMPL,
implementing AI capabilities for the KeaBuilder platform (funnels, lead capture,
automation).

The whole submission is one FastAPI app. Every form question maps to a
self-contained folder with code, docs, and a sample output. The reviewer
should not have to hunt — start at the **Question Map** below.

---

## Question Map

| Q | Topic                          | Code                                  | Docs                                              | Sample Output                                |
|---|--------------------------------|---------------------------------------|---------------------------------------------------|----------------------------------------------|
| 1 | Lead classification + response | `app/q1_classifier/`                  | [§Q1 below](#q1--lead-classification--response)   | `samples/q1_classify_*.json`                 |
| 2 | Multi-provider routing         | `app/q2_routing/`                     | `app/q2_routing/ARCHITECTURE.md`                  | `samples/q2_route_*.json`                    |
| 3 | LoRA integration               | `app/q3_lora/snippet.py`              | `app/q3_lora/LORA.md`                             | (write-up)                                   |
| 4 | Similarity (text + face)       | `app/q4_similarity/`                  | [§Q4 below](#q4--similarity-search-text--face)   | `samples/q4_*_search_*.json`                 |
| 5 | Fallback / error handling      | `app/q5_resilience/` (live in Q1+Q2)  | `app/q5_resilience/RESILIENCE.md`                 | (visible in Q1/Q2 logs + Q2 fallback sample) |
| 6 | High-volume scaling design     | —                                     | `app/q6_scaling/SCALING.md`                       | (mermaid diagram in doc)                     |
| 7 | Tools & experience             | (answered in the Google Form text field) | —                                              | —                                            |

---

## 60-second quickstart

```bash
git clone <this-repo> && cd assessment
python -m venv .venv
.venv\Scripts\activate           # Windows
# or: source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
cp .env.example .env             # then edit OPENAI_API_KEY (or leave MOCK_OPENAI=1 for keyless run)
uvicorn app.main:app --port 8000
```

Open `http://localhost:8000/docs` for the auto-generated Swagger UI.

### Run keyless (no OpenAI key needed)

```bash
MOCK_OPENAI=1 uvicorn app.main:app --port 8000
```

The Q1 classifier falls back to a deterministic heuristic so every endpoint
still works end-to-end. Mock outputs are tagged `"source": "mock"` so the
distinction is explicit.

### Regenerate the cached sample outputs

```bash
python -m scripts.generate_samples           # uses real OpenAI
MOCK_OPENAI=1 python -m scripts.generate_samples   # keyless
```

---

## Q1 — Lead Classification & Response

> Classify each lead as hot/warm/cold. Generate a personalized first-touch
> reply. Handle incomplete inputs gracefully.

### Pipeline

```
LeadInput  --(classifier prompt, JSON-mode)-->  Classification
                                                {category, confidence,
                                                 reasoning, missing_signals}
            --(response prompt, conditioned on classification)-->  Reply
```

Both LLM calls are wrapped with the Q5 retry decorator. On schema-violation
or transient API failure, retry; on permanent failure, return 502/503 with
a `Retry-After` header.

### Endpoint

```bash
curl -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Priya Mehta",
    "company": "NorthStar Realty",
    "message": "Ready to roll out funnels for 8 agents this month. Pricing?",
    "budget": "$3-5k/mo",
    "timeline": "this month"
  }'
```

### Example output

```json
{
  "classification": {
    "category": "hot",
    "confidence": 0.92,
    "reasoning": "Explicit buying intent ('ready to roll out'), stated budget and team size, named timeline.",
    "missing_signals": []
  },
  "suggested_reply": "Hi Priya, thanks for the note. Rolling out for 8 agents this month is exactly the kind of setup KeaBuilder is built for — happy to walk you through the multi-agent funnel templates live. I have Tuesday 10am or Thursday 2pm open. Which works? Sam from KeaBuilder",
  "source": "openai"
}
```

### Incomplete-input handling

The classifier is instructed to default to `cold` when critical fields are
missing, surface them in `missing_signals`, and the response generator falls
back to a qualifying question instead of inventing facts. See:
`samples/q1_classify_incomplete.json`.

---

## Q2 — Multi-Provider Routing

See `app/q2_routing/ARCHITECTURE.md` for the diagram + frontend↔backend
interaction model. Quick endpoint demo:

```bash
# normal path
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"modality":"image","prompt":"modern real-estate hero banner"}'

# force fallback chain to run
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"modality":"image","prompt":"hero banner","force_fail":true}'
```

The first call returns `"provider": "openai-dalle3"`. The second forces the
primary to fail and returns `"provider": "stability-sdxl", "fallback_used": true`.
Sample: `samples/q2_route_image_fallback.json`.

---

## Q4 — Similarity Search (text + face)

### Text — sentence-transformers MiniLM + cosine

```bash
curl -X POST http://localhost:8000/search/text \
  -H 'Content-Type: application/json' \
  -d '{"query":"I run a real estate firm and want to capture inbound leads","top_k":3}'
```

Sample: `samples/q4_text_search_real_estate.json`. Top hit is the matching
real-estate lead in the corpus with a cosine score of ~0.59. Corpus lives
at `app/q4_similarity/data/texts.json` (20 sample leads).

### Face — insightface buffalo_l + cosine

```bash
curl -X POST 'http://localhost:8000/search/face?top_k=3' \
  -F 'file=@app/q4_similarity/data/faces/face_1.jpg'
```

Sample: `samples/q4_face_search_face_1.json`. The query face matches itself
at score 1.0; other faces in the corpus score <0.2 (they are different
synthetic identities), confirming the embedding distinguishes between them.

### Why no FAISS

The corpus is <1k vectors. `numpy.dot` on L2-normalized embeddings is fast
enough and saves a non-trivial dependency. For >100k vectors I'd switch to
FAISS or a vector DB (pgvector, Qdrant); the function signature wouldn't
change.

---

## Layout

```
.
├── app/
│   ├── main.py                       # FastAPI; mounts q1, q2, q4 routers
│   ├── config.py                     # env loader; MOCK_OPENAI flag
│   ├── q1_classifier/                # Q1 — classifier + response
│   ├── q2_routing/                   # Q2 — multi-provider routing
│   ├── q3_lora/                      # Q3 — LoRA write-up + snippet
│   ├── q4_similarity/                # Q4 — text + face search
│   ├── q5_resilience/                # Q5 — retry + circuit breaker
│   └── q6_scaling/                   # Q6 — scaling design
├── samples/                          # cached real outputs (committed)
├── scripts/generate_samples.py       # regenerate /samples
└── tests/                            # pytest (pure-function only)
```

---

## What's deliberately *not* here

- **No actual image / video / voice generation** — the providers are mocked
  to demonstrate the routing architecture. Plugging in real providers is
  one new class per provider; see `app/q2_routing/base.py`.
- **No LoRA training run** — the integration code path is documented and
  referenced. Training would be a separate 24h project.
- **No FAISS / vector DB / k8s** — added complexity that doesn't change the
  question's answer at this corpus size.
- **No frontend** — the spec asks about backend architecture and APIs.

These are honest tradeoffs given the 24-hour window. See each Q's docs for
how production would extend.
