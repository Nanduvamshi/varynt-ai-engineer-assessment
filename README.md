# VARYNT — AI Engineer Assessment

FastAPI app implementing AI capabilities (lead classification, multi-provider
generation routing, similarity search, resilience layer) for the KeaBuilder
platform.

---

## Quickstart

```bash
git clone https://github.com/Nanduvamshi/varynt-ai-engineer-assessment.git
cd varynt-ai-engineer-assessment

python -m venv .venv
.venv\Scripts\activate           # Windows
# or: source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
copy .env.example .env           # Windows  (cp on macOS/Linux)
# edit .env and set OPENAI_API_KEY=sk-...   (or leave MOCK_OPENAI=1 for keyless)

uvicorn app.main:app --port 8000
```

Open http://localhost:8000/docs for the auto-generated Swagger UI.

### Run keyless (no OpenAI key needed)

Set `MOCK_OPENAI=1` in `.env`. The classifier falls back to a deterministic
heuristic. Mocked outputs are tagged `"source": "mock"` in responses so the
distinction is explicit.

---

## Project layout

```
.
├── app/
│   ├── main.py                    # FastAPI; mounts all routers
│   ├── config.py                  # env loader; MOCK_OPENAI flag
│   ├── q1_classifier/             # POST /classify
│   ├── q2_routing/                # POST /generate (image/video/voice)
│   ├── q3_lora/                   # LoRA integration reference snippet
│   ├── q4_similarity/             # POST /search/text, /search/face
│   └── q5_resilience/             # retry decorator + circuit breaker
├── samples/                       # cached real outputs from the endpoints
├── scripts/generate_samples.py    # regenerate /samples by hitting endpoints
└── tests/                         # pytest suite
```

---

## Endpoints

```bash
# Lead classifier
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"name":"Priya","company":"NorthStar","message":"Ready to roll out funnels for 8 agents this month","budget":"$3-5k/mo","timeline":"this month"}'

# Multi-provider generation (image / video / voice)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"modality":"image","prompt":"hero banner"}'

# Same endpoint with force_fail=true to demo the fallback chain
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"modality":"image","prompt":"hero banner","force_fail":true}'

# Text similarity search
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query":"yoga studio class booking","top_k":3}'

# Face similarity search
curl -X POST "http://localhost:8000/search/face?top_k=3" \
  -F "file=@app/q4_similarity/data/faces/face_1.jpg"
```

---

## Sample outputs

`samples/` contains real outputs produced by running each endpoint.
Regenerate them with:

```bash
python -m scripts.generate_samples
```

---

## Tests

```bash
python -m pytest -q
```

11 tests covering the classifier, routing, fallback chain, and similarity search.
