from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import students, teachback, teacher, topics
from .seed import seed_db

app = FastAPI(title="TeachBack", version="1.0.0",
              description="Teach-back based learning state estimation (NLP + HMM)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(students.router)
app.include_router(topics.router)
app.include_router(teachback.router)
app.include_router(teacher.router)


@app.on_event("startup")
def startup():
    seed_db()


@app.get("/api/health")
def health():
    from .hmm.model import hmm_available

    return {"status": "ok", "hmm_trained": hmm_available()}
