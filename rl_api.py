"""
RL Agent Inference & Training API Endpoints

Provides:
  POST /agent/action - Get agent decision for resume screening
  POST /agent/train - Train agent (curriculum learning)
  POST /agent/learn - Single step learning
  GET /agent/stats - Get agent performance stats
  POST /agent/save-checkpoint - Save agent state
  POST /agent/load-checkpoint - Load agent state
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from rl_agent import ResumeScreeningAgent, Experience
from agent_trainer import AgentTrainer, create_demo_data
from models import AgentOutput


# Global agent instances
_agent: Optional[ResumeScreeningAgent] = None
_trainer: Optional[AgentTrainer] = None


class AgentActionRequest(BaseModel):
    resume_text: str = Field(..., min_length=1)
    job_description: str = Field(..., min_length=1)
    task_difficulty: str = Field(default="medium", pattern="^(easy|medium|hard)$")
    task_id: str = Field(default="task_1")
    use_exploration: bool = Field(default=False)


class AgentActionResponse(BaseModel):
    id: str
    decision: str
    score: float
    reasoning: str
    confidence: float
    state_features: Dict[str, float]
    epsilon: float
    task_difficulty: str


class AgentStatsResponse(BaseModel):
    total_episodes: int
    avg_reward_100: float
    epsilon: float
    task_performance: Dict[str, float]
    buffer_size: int


class TrainingRequest(BaseModel):
    task_id: str = Field(default="task_1")
    task_difficulty: str = Field(default="medium")
    num_episodes: int = Field(default=10, ge=1, le=100)
    learning_rate: Optional[float] = Field(default=None)
    use_demo_data: bool = Field(default=True)


class CheckpointRequest(BaseModel):
    filepath: str = Field(..., min_length=1)


def get_agent() -> ResumeScreeningAgent:
    """Get or create global agent instance"""
    global _agent
    if _agent is None:
        _agent = ResumeScreeningAgent()
        # Load trained checkpoint if it exists (RL environment requires loaded Q-values)
        import os
        checkpoint_path = "agent_checkpoint.json"
        if os.path.exists(checkpoint_path):
            try:
                _agent.load_checkpoint(checkpoint_path)
                print(f"✓ Loaded agent checkpoint from {checkpoint_path}")
            except Exception as e:
                print(f"⚠ Could not load checkpoint: {e}")
    return _agent


def get_trainer() -> AgentTrainer:
    """Get or create global trainer instance"""
    global _trainer
    if _trainer is None:
        _trainer = AgentTrainer(get_agent())
    return _trainer


# ============================================================================
# API FUNCTIONS
# ============================================================================

def agent_action(request: AgentActionRequest) -> AgentActionResponse:
    """
    Get agent's action/decision for a resume-JD pair
    """
    import uuid
    from datetime import datetime
    
    agent = get_agent()
    
    # Generate unique analysis ID
    analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    # Extract state
    past_reward_avg = (
        sum(agent.episode_rewards[-100:]) / len(agent.episode_rewards[-100:])
        if agent.episode_rewards 
        else 0.5
    )
    
    state = agent.extract_state(
        request.resume_text,
        request.job_description,
        past_reward_avg
    )
    
    # Choose action
    # Use epsilon=0 for pure exploitation during inference (no exploration)
    # Use epsilon=1.0 only if explicitly requesting exploration
    epsilon_override = 1.0 if request.use_exploration else 0.0
    decision, score, reasoning = agent.choose_action(
        state,
        task_difficulty=request.task_difficulty,
        epsilon=epsilon_override
    )
    
    # Extract state components
    _, _, growth_score, reward_avg, confidence = state
    
    return AgentActionResponse(
        id=analysis_id,
        decision=decision,
        score=round(score, 4),
        reasoning=reasoning,
        confidence=round(float(confidence), 4),
        state_features={
            "keyword_similarity": round(float(state[0]), 4),
            "skills_similarity": round(float(state[1]), 4),
            "growth_score": round(float(growth_score), 4),
            "avg_reward": round(float(reward_avg), 4)
        },
        epsilon=round(agent.epsilon, 4),
        task_difficulty=request.task_difficulty
    )


def agent_learn(
    resume_text: str,
    job_description: str,
    agent_action: tuple,
    reward: float,
    task_difficulty: str
) -> Dict[str, Any]:
    """
    Single learning step for agent
    """
    agent = get_agent()
    
    # Create experience
    past_reward_avg = sum(agent.episode_rewards[-100:]) / len(agent.episode_rewards[-100:]) if agent.episode_rewards else 0.5
    state = agent.extract_state(resume_text, job_description, past_reward_avg)
    next_state = agent.extract_state(resume_text, job_description, reward)
    
    experience = Experience(
        state=state,
        action=agent_action,
        reward=reward,
        next_state=next_state,
        done=True
    )
    
    td_error = agent.learn(experience)
    agent.update_episode_stats(reward, task_difficulty)
    
    return {
        "td_error": float(td_error),
        "epsilon": float(agent.epsilon),
        "buffer_size": len(agent.replay_buffer),
        "total_episodes": len(agent.episode_rewards)
    }


def agent_train_curriculum() -> Dict[str, Any]:
    """
    Run full curriculum training (Easy → Medium → Hard)
    """
    trainer = get_trainer()
    task1, task2, task3 = create_demo_data()
    
    summary = trainer.run_curriculum(
        task1_data=task1,
        task2_data=task2,
        task3_data=task3,
        episodes_per_task=15
    )
    
    return summary


def agent_stats() -> AgentStatsResponse:
    """
    Get agent performance statistics
    """
    agent = get_agent()
    perf = agent.get_performance_summary()
    
    return AgentStatsResponse(
        total_episodes=perf["total_episodes"],
        avg_reward_100=perf["avg_reward_100"],
        epsilon=round(perf["epsilon"], 4),
        task_performance={
            k: round(v, 4) 
            for k, v in perf["task_performance"].items()
        },
        buffer_size=perf["buffer_size"]
    )


def agent_save_checkpoint(request: CheckpointRequest) -> Dict[str, str]:
    """
    Save agent weights and state
    """
    agent = get_agent()
    agent.save_checkpoint(request.filepath)
    
    return {
        "status": "saved",
        "filepath": request.filepath,
        "episodes": str(len(agent.episode_rewards))
    }


def agent_load_checkpoint(request: CheckpointRequest) -> Dict[str, str]:
    """
    Load agent weights and state
    """
    agent = get_agent()
    agent.load_checkpoint(request.filepath)
    
    return {
        "status": "loaded",
        "filepath": request.filepath,
        "episodes": str(len(agent.episode_rewards))
    }


def agent_reset() -> Dict[str, str]:
    """
    Reset agent to initial state
    """
    global _agent, _trainer
    _agent = ResumeScreeningAgent()
    _trainer = AgentTrainer(_agent)
    
    return {
        "status": "reset",
        "message": "Agent and trainer reset to initial state"
    }
