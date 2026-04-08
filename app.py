from __future__ import annotations

import os
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server import create_fastapi_app
from fastapi.middleware.cors import CORSMiddleware

from models import BaselineRequest, BaselineResponse, GraderRequest, GraderResponse, TaskSpec
from server.environment import ResumeEnv
from server.grader import grade_submission
from server.tasks import get_tasks
from server.rl_api import (
    AgentActionRequest, AgentActionResponse, AgentStatsResponse,
    TrainingRequest, CheckpointRequest,
    agent_action, agent_train_curriculum, agent_stats,
    agent_save_checkpoint, agent_load_checkpoint, agent_reset
)


app = create_fastapi_app(ResumeEnv)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ... CORS middleware ...
# Path to UI
ui_path = os.path.join(os.path.dirname(__file__), "..", "UI_dist")

# Mount static files
if os.path.exists(ui_path):
    app.mount("/static", StaticFiles(directory=ui_path), name="static")

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    index_file = os.path.join(ui_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"error": "Not found"}

@app.get("/")
def root() -> dict[str, object]:
    return {
        "name": "Resume Screening & Ranking with RL Agent",
        "status": "ok",
        "version": "2.0",
        "endpoints": [
            "/reset", "/step", "/state", "/tasks", "/grader", "/baseline",
            "/agent/action", "/agent/train", "/agent/stats", "/agent/checkpoint",
            "/docs"
        ],
    }


@app.get("/tasks", response_model=list[TaskSpec])
def tasks() -> list[TaskSpec]:
    return get_tasks()


@app.post("/grader", response_model=GraderResponse)
def grader(request: GraderRequest) -> GraderResponse:
    return grade_submission(request)


# ============================================================================
# RL AGENT ENDPOINTS
# ============================================================================

@app.options("/agent/action")
async def options_agent_action():
    return {}

@app.post("/agent/action", response_model=AgentActionResponse)
def agent_get_action(request: AgentActionRequest) -> AgentActionResponse:
    """
    Get RL agent's decision for a resume-JD pair.
    
    Returns decision, score, reasoning, and state features.
    """
    return agent_action(request)


@app.post("/agent/train", response_model=dict)
def agent_train_endpoint(request: TrainingRequest) -> dict:
    """
    Train agent using curriculum learning.
    Runs: Easy (keyword) → Medium (skills) → Hard (growth)
    """
    return agent_train_curriculum()


@app.get("/agent/stats", response_model=AgentStatsResponse)
def agent_get_stats() -> AgentStatsResponse:
    """
    Get agent performance statistics and learning curves.
    """
    return agent_stats()


@app.post("/agent/checkpoint/save", response_model=dict)
def agent_save(request: CheckpointRequest) -> dict:
    """
    Save agent weights and training state to file.
    """
    return agent_save_checkpoint(request)


@app.post("/agent/checkpoint/load", response_model=dict)
def agent_load(request: CheckpointRequest) -> dict:
    """
    Load agent weights and training state from file.
    """
    return agent_load_checkpoint(request)


@app.post("/agent/reset", response_model=dict)
def agent_reset_endpoint() -> dict:
    """
    Reset agent to initial untrained state.
    """
    return agent_reset()
