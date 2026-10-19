import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import time
from dotenv import load_dotenv

load_dotenv()

from agent.graph import run_query, get_agent, clear_session
from agent.cache import cache_stats, clear as clear_cache
from tools.rag_tool import _get_collection
from tools.graph_tool import _get_driver
from tools.counts import ranked_counts

app = FastAPI(
    title="Recognition Analytics Agent",
    description="Production HR recognition analytics chatbot",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question:   str
    session_id: str = "default"


class QueryResponse(BaseModel):
    answer:               str
    time_taken_seconds:   float
    route_used:           Optional[str]       = None
    hallucination_flag:   Optional[bool]      = None
    hallucination_score:  Optional[float]     = None
    hallucinated_claims:  Optional[List[str]] = None
    was_refused:          Optional[bool]      = None
    grader_error:         Optional[bool]      = None
    trace_id:             Optional[str]       = None
    cached:               Optional[bool]      = None
    model_tier:           Optional[str]       = None
    escalated:            Optional[bool]      = None


class ClearSessionRequest(BaseModel):
    session_id: str = "default"


class HealthResponse(BaseModel):
    status:  str
    message: str


@app.get("/", response_model=HealthResponse)
def root():
    return {"status": "ok", "message": "Recognition Analytics Agent v4 (production) is running."}


@app.get("/health", response_model=HealthResponse)
def health():
    neo4j_status = "connected" if _get_driver() else "unavailable (NetworkX fallback active)"
    return {"status": "ok", "message": f"All systems operational. Neo4j: {neo4j_status}."}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start = time.time()
    try:
        result = run_query(request.question, session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
    elapsed = round(time.time() - start, 2)

    return QueryResponse(
        answer=result["answer"],
        time_taken_seconds=elapsed,
        route_used=result.get("route_used"),
        hallucination_flag=result.get("hallucination_flag"),
        hallucination_score=result.get("hallucination_score"),
        hallucinated_claims=result.get("hallucinated_claims", []),
        was_refused=result.get("was_refused", False),
        grader_error=result.get("grader_error", False),
        trace_id=result.get("trace_id"),
        cached=result.get("cached", False),
        model_tier=result.get("model_tier"),
        escalated=result.get("escalated", False),
    )


@app.post("/session/clear")
def session_clear(request: ClearSessionRequest):
    clear_session(request.session_id)
    return {"status": "ok", "message": f"Session '{request.session_id}' cleared."}


@app.get("/cache/stats")
def get_cache_stats():
    """Hit/miss counters since the server started, entry counts, and the code/data versions in the cache keys."""
    return cache_stats()


@app.post("/cache/clear")
def post_cache_clear(which: str = "all"):
    """which = llm | answers | all"""
    if which not in ("llm", "answers", "all"):
        raise HTTPException(status_code=400, detail="which must be llm, answers or all")
    clear_cache(which)
    return {"status": "ok", "message": f"Cleared {which} cache."}


@app.post("/index/build")
def build_index():
    """Trigger ChromaDB index build (runs once, reuses after)."""
    try:
        _get_collection()
        return {"status": "ok", "message": "ChromaDB index ready."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dataset/summary")
def dataset_summary():
    try:
        import pandas as pd
        data_path = os.getenv("DATA_PATH", "./data/annotated_results.csv")
        df        = pd.read_csv(data_path)
        return {
            "total_rows":              len(df),
            "categories":              ranked_counts(df["category_name"]).to_dict(),
            "subcategory_count":       int(df["subcategory_name"].nunique()),
            "unique_recipient_titles": int(df["recipient_title"].nunique()),
            "unique_award_titles":     int(df["award_title"].nunique()),
            "neo4j_available":         _get_driver() is not None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
