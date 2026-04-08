from __future__ import annotations

from typing import List

from models import TaskSpec


def get_tasks() -> List[TaskSpec]:
    return [
        TaskSpec(
            task_id="easy_keyword_match",
            difficulty="easy",
            title="Keyword Match Screening",
            description="Screen a resume by comparing direct keyword overlap against the job description.",
            evaluation_focus=["keyword extraction", "requirement matching", "clear shortlist threshold"],
        ),
        TaskSpec(
            task_id="medium_transferable_skills",
            difficulty="medium",
            title="Transferable Skills Expansion",
            description="Infer adjacent skills from a lightweight skill graph before scoring candidate fit.",
            evaluation_focus=["skill expansion", "partial credit", "reasoning about transferable skills"],
        ),
        TaskSpec(
            task_id="hard_growth_pattern",
            difficulty="hard",
            title="Growth Pattern Analysis",
            description="Estimate future potential from progression signals such as promotions, increasing scope, and leadership.",
            evaluation_focus=["career trajectory", "growth heuristics", "balanced decision making"],
        ),
    ]
