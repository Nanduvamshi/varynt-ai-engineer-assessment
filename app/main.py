from fastapi import FastAPI
from app.q1_classifier.router import router as classifier_router
from app.q2_routing.router import router as routing_router
from app.q4_similarity.router import router as similarity_router

app = FastAPI(
    title="VARYNT AI Engineer Assessment",
    description="KeaBuilder AI capabilities: lead scoring, multi-provider generation, similarity search.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "service": "varynt-ai-assessment",
        "endpoints": [
            "POST /classify",
            "POST /generate",
            "POST /search/text",
            "POST /search/face",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(classifier_router)
app.include_router(routing_router)
app.include_router(similarity_router)
