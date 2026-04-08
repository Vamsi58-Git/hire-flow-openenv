#!/usr/bin/env python3
"""
OpenEnv Inference Script for HireFlow Resume Screening Agent
Outputs [START]/[STEP]/[END] logs in OpenEnv-compliant format.
"""

import os
import sys
import requests
from typing import Dict, Tuple

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
HF_TOKEN = os.getenv("HF_TOKEN", "")


def get_test_cases() -> Dict:
    """Return 3 test cases: easy, medium, hard"""
    return {
        "easy_keyword_match": [{
            "task_id": "easy_keyword_match",
            "resume": "John Smith\nSenior Software Engineer | 7 Years\nEXPERIENCE:\nSenior Backend Engineer at TechCorp (2021-Present)\nLed development of microservices using Python and Django\nManaged team of 4 junior developers\nTech Stack: Python, Django, PostgreSQL, Redis, Docker, Kubernetes\nSKILLS: Python (Expert), Django (Advanced), PostgreSQL (Advanced), Docker (Intermediate)",
            "job_description": "Senior Python Developer - Backend Team\nExperience Required: 5+ years\nRequired Skills:\n5+ years Python development\nDjango or FastAPI experience\nPostgreSQL experience\nMicroservices architecture\nDocker containerization",
            "expected_decision": "SHORTLIST",
            "expected_score_range": (0.85, 1.0),
        }],
        "medium_transferable_skills": [{
            "task_id": "medium_transferable_skills",
            "resume": "Maria Rodriguez\nBackend Developer | 4 Years\nEXPERIENCE:\nBackend Engineer at FinanceApp (2021-Present)\nDeveloped microservices using Java Spring Boot\nDesigned and optimized PostgreSQL databases\nImplemented REST APIs for payment systems\nMentored 2 junior developers\nTech Stack: Java, Spring Boot, PostgreSQL, Docker\nSKILLS: Java (Advanced), Spring Boot (Advanced), PostgreSQL (Advanced), REST APIs (Advanced), Docker (Intermediate)",
            "job_description": "Senior Python Developer - Backend Team\nExperience Required: 5+ years\nRequired Skills:\n5+ years Python development\nDjango or FastAPI experience\nPostgreSQL experience\nMicroservices architecture\nDocker containerization",
            "expected_decision": "SHORTLIST",
            "expected_score_range": (0.55, 0.70),
        }],
        "hard_growth_pattern": [{
            "task_id": "hard_growth_pattern",
            "resume": "Alex Chen\nJunior JavaScript Developer | 2 Years\nEXPERIENCE:\nJunior Frontend Developer at WebAgency (2022-Present)\nDeveloped responsive UIs using React and JavaScript\nFixed bugs and implemented features\nWrote Jest unit tests\nTech Stack: React, JavaScript, HTML/CSS, Tailwind\nSKILLS: JavaScript (Intermediate), React (Intermediate), HTML/CSS (Intermediate), Node.js (Beginner)",
            "job_description": "Senior Python Developer - Backend Team\nExperience Required: 5+ years\nRequired Skills:\n5+ years Python development\nDjango or FastAPI experience\nPostgreSQL experience\nMicroservices architecture\nDocker containerization",
            "expected_decision": "REJECT",
            "expected_score_range": (0.0, 0.25),
        }],
    }


from openai import OpenAI

# After line 13, add:
openai_client = OpenAI(api_key=HF_TOKEN, base_url=f"{API_BASE_URL}/v1")

# Replace call_agent_action():
def call_agent_action(resume: str, job_description: str) -> Dict:
    """Call backend /agent/action endpoint using OpenAI client"""
    try:
        url = f"{API_BASE_URL}/agent/action"
        # Use OpenAI client for the request
        response = openai_client.with_raw_response.post(
            url,
            json={"resume": resume, "job_description": job_description},
        )
        # ... rest of code
        if response.status_code == 200:
            result = response.json()
            return {
                "decision": result.get("decision", "UNKNOWN"),
                "score": float(result.get("score", 0.5)),
            }
        return {"decision": "UNKNOWN", "score": 0.0}
    except Exception as e:
        return {"decision": "UNKNOWN", "score": 0.0}


def compute_reward(decision: str, expected_decision: str, score: float, expected_range: Tuple) -> float:
    """Decision correct: +0.5, Score in range: +0.5, Max: 1.0"""
    reward = 0.0
    if decision == expected_decision:
        reward += 0.5
    if expected_range[0] <= score <= expected_range[1]:
        reward += 0.5
    return min(reward, 1.0)


def run_baseline(env_name: str = "hireflow-resume-screening"):
    """Run with OpenEnv-compliant [START]/[STEP]/[END] logging"""
    test_cases = get_test_cases()
    all_rewards = []

    for difficulty, cases in test_cases.items():
        for case in cases:
            print(f"[START] task={case['task_id']} env={env_name} model={MODEL_NAME}", flush=True)
            
            result = call_agent_action(case["resume"], case["job_description"])
            decision = result["decision"]
            score = result["score"]
            
            reward = compute_reward(decision, case["expected_decision"], score, case["expected_score_range"])
            all_rewards.append(reward)
            
            print(f"[STEP] step=1 action=decision={decision} score={score:.2f} reward={reward:.2f} done=true error=null", flush=True)
            
            success = (decision == case["expected_decision"] and case["expected_score_range"][0] <= score <= case["expected_score_range"][1])
            print(f"[END] success={str(success).lower()} steps=1 score={score:.2f} rewards={reward:.2f}", flush=True)
            print("", flush=True)

    return all_rewards


def main():
    try:
        print("HireFlow Inference Script", file=sys.stderr)
        print(f"API_BASE_URL: {API_BASE_URL}", file=sys.stderr)
        print(f"MODEL_NAME: {MODEL_NAME}", file=sys.stderr)
        
        rewards = run_baseline()
        if rewards:
            avg = sum(rewards) / len(rewards)
            print(f"Average Reward: {avg:.2f}", file=sys.stderr)
            sys.exit(0 if avg >= 0.5 else 1)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
