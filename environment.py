from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from openenv.core.env_server import Environment

from models import AgentOutput
from server.bias import detect_bias
from server.skill_graph import expand_skills


STOP_WORDS = {
    "and",
    "the",
    "with",
    "for",
    "from",
    "that",
    "this",
    "into",
    "will",
    "have",
    "has",
    "using",
    "years",
    "year",
    "experience",
    "skills",
    "role",
    "team",
    "work",
}


@dataclass
class HeuristicResult:
    keyword_score: float
    transferable_score: float
    growth_score: float
    final_score: float
    matched_keywords: List[str]
    expanded_resume_skills: List[str]
    growth_signals: List[str]


class ResumeEnv(Environment):
    max_steps: int = 5

    def __init__(self) -> None:
        self.resume_text = ""
        self.job_description = ""
        self.steps_used = 0
        self.last_output: Dict[str, Any] | None = None
        self.last_reward = 0.0
        self.done = False

    def reset(self, **kwargs: Any) -> Dict[str, Any]:
        self.resume_text = kwargs.get("resume_text", "")
        self.job_description = kwargs.get("job_description", "")
        self.steps_used = 0
        self.last_output = None
        self.last_reward = 0.0
        self.done = False
        return self.state

    def step(self, action: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        if kwargs.get("resume_text"):
            self.resume_text = kwargs["resume_text"]
        if kwargs.get("job_description"):
            self.job_description = kwargs["job_description"]

        self.steps_used += 1

        # ============================================================
        # STEP 1: Extract user input (ALWAYS from action, never defaults)
        # ============================================================
        if hasattr(action, "decision"):
            # action is a Pydantic model (AgentOutput)
            user_decision = action.decision
            user_score = action.score
            user_reasoning = action.reasoning
        else:
            # action is a dict
            user_decision = action.get("decision")
            user_score = action.get("score")
            user_reasoning = action.get("reasoning")

        # ============================================================
        # STEP 2: Validate and normalize user input
        # ============================================================
        # Ensure these are valid types (but preserve user values)
        decision = str(user_decision).strip() if user_decision else "reject"
        
        try:
            score = float(user_score) if user_score is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        
        reasoning = str(user_reasoning).strip() if user_reasoning else ""

        # ============================================================
        # STEP 3: Create submission for reward calculation
        # ============================================================
        submission = AgentOutput(
            decision=decision,
            score=score,
            reasoning=reasoning if reasoning else "No reasoning provided.",
        )
        
        # ============================================================
        # STEP 4: Get system's expected output (for reward ONLY)
        # ============================================================
        expected_output, _ = self.build_rule_based_output(self.resume_text, self.job_description)

        # ============================================================
        # STEP 5: Calculate reward based on comparison
        # ============================================================
        correct_decision = 0.5 if submission.decision == expected_output.decision else 0.0
        score_accuracy = self.score_accuracy(expected_output.score, submission.score)
        reasoning_quality = self.reasoning_quality(submission.reasoning, self.resume_text, self.job_description)
        bias_penalty = detect_bias(submission.reasoning).penalty
        overflow_penalty = 0.5 if self.steps_used > self.max_steps else 0.0

        reward = round(
            correct_decision + score_accuracy + reasoning_quality - bias_penalty - overflow_penalty,
            4,
        )
        
        # ============================================================
        # STEP 6: Update environment state
        # ============================================================
        self.done = self.steps_used >= self.max_steps
        self.last_reward = reward
        
        # ============================================================
        # STEP 7: STORE USER'S DECISION (NOT system's expected output)
        # =========================================================
        # This is the ONLY place last_output gets set in step()
        # It stores the USER's input, not the system's expected output
        # NO OVERWRITES AFTER THIS POINT
        # =========================================================
        self.last_output = {
            "decision": decision,
            "score": score,
            "reasoning": reasoning if reasoning else "No reasoning provided.",
        }

        return {
            "state": self.state,
            "reward": reward,
            "done": self.done,
            "info": {
                "expected_decision": expected_output.decision,
                "expected_score": expected_output.score,
                "steps_used": self.steps_used,
            },
        }

    @property
    def state(self) -> Dict[str, Any]:
        return {
            "resume_text": self.resume_text,
            "job_description": self.job_description,
            "steps_used": self.steps_used,
            "steps_remaining": max(0, self.max_steps - self.steps_used),
            "last_output": self.last_output,
            "last_reward": self.last_reward,
            "done": self.done,
        }

    @staticmethod
    def tokenize(text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z\+\#]{1,}", text.lower())
        return [token for token in tokens if token not in STOP_WORDS]

    @staticmethod
    def extract_phrases(text: str) -> List[str]:
        phrases = re.findall(r"\b(?:[A-Za-z]+\s){0,2}[A-Za-z]+\b", text.lower())
        cleaned = {" ".join(part for part in phrase.split() if part not in STOP_WORDS).strip() for phrase in phrases}
        return sorted(item for item in cleaned if item)

    def _keyword_match(self, resume_text: str, job_description: str) -> Tuple[float, List[str]]:
        resume_terms = set(self.tokenize(resume_text))
        jd_terms = set(self.tokenize(job_description))
        matched = sorted(resume_terms & jd_terms)
        if not jd_terms:
            return 0.0, matched
        return min(1.0, len(matched) / max(1, len(jd_terms))), matched

    def _transferable_skill_score(self, resume_text: str, job_description: str) -> Tuple[float, List[str]]:
        resume_skills = set(self.tokenize(resume_text)) | set(self.extract_phrases(resume_text))
        jd_skills = set(self.tokenize(job_description)) | set(self.extract_phrases(job_description))
        expanded = set(expand_skills(resume_skills))
        overlap = sorted(expanded & jd_skills)
        if not jd_skills:
            return 0.0, sorted(expanded)
        return min(1.0, len(overlap) / max(1, len(jd_skills))), sorted(expanded)

    def _growth_pattern_score(self, resume_text: str) -> Tuple[float, List[str]]:
        lowered = resume_text.lower()
        signals = []

        signal_map = {
            "promotion": [r"\bpromoted\b", r"\bpromotion\b"],
            "leadership": [r"\bled\b", r"\bmanaged\b", r"\bmentored\b"],
            "ownership": [r"\bowned\b", r"\bdelivered\b", r"\bbuilt\b"],
            "scope increase": [r"\bsenior\b", r"\blead\b", r"\bprincipal\b"],
            "impact": [r"\bincreased\b", r"\breduced\b", r"\bimproved\b", r"\bgrew\b"],
        }

        for label, patterns in signal_map.items():
            if any(re.search(pattern, lowered) for pattern in patterns):
                signals.append(label)

        years = [int(value) for value in re.findall(r"(\d+)\+?\s+years", lowered)]
        if years and max(years) >= 3:
            signals.append("sustained experience")

        score = min(1.0, len(set(signals)) / 5.0)
        return score, sorted(set(signals))

    def assess(self, resume_text: str, job_description: str) -> HeuristicResult:
        keyword_score, matched_keywords = self._keyword_match(resume_text, job_description)
        transferable_score, expanded_resume_skills = self._transferable_skill_score(resume_text, job_description)
        growth_score, growth_signals = self._growth_pattern_score(resume_text)

        final_score = (0.45 * keyword_score) + (0.30 * transferable_score) + (0.25 * growth_score)

        return HeuristicResult(
            keyword_score=round(keyword_score, 4),
            transferable_score=round(transferable_score, 4),
            growth_score=round(growth_score, 4),
            final_score=round(min(1.0, max(0.0, final_score)), 4),
            matched_keywords=matched_keywords,
            expanded_resume_skills=expanded_resume_skills,
            growth_signals=growth_signals,
        )

    def build_rule_based_output(self, resume_text: str, job_description: str) -> Tuple[AgentOutput, HeuristicResult]:
        result = self.assess(resume_text, job_description)
        decision = "shortlist" if result.final_score >= 0.6 else "reject"

        matched = ", ".join(result.matched_keywords[:8]) or "limited direct matches"
        growth = ", ".join(result.growth_signals) or "no strong growth signals"
        reasoning = (
            f"Decision based on keyword overlap ({result.keyword_score:.2f}), "
            f"transferable skill coverage ({result.transferable_score:.2f}), "
            f"and growth signals ({result.growth_score:.2f}). "
            f"Matched skills: {matched}. Growth indicators: {growth}."
        )

        return AgentOutput(decision=decision, score=result.final_score, reasoning=reasoning), result

    @staticmethod
    def reasoning_quality(reasoning: str, resume_text: str, job_description: str) -> float:
        score = 0.0
        if len(reasoning.split()) >= 12:
            score += 0.08
        if any(token in reasoning.lower() for token in ["keyword", "skill", "growth", "experience"]):
            score += 0.06
        shared_terms = set(re.findall(r"\b[a-z]{4,}\b", resume_text.lower())) & set(
            re.findall(r"\b[a-z]{4,}\b", job_description.lower())
        )
        if shared_terms and any(term in reasoning.lower() for term in list(shared_terms)[:5]):
            score += 0.06
        return round(min(0.2, score), 4)

    @staticmethod
    def score_accuracy(expected: float, actual: float) -> float:
        distance = abs(expected - actual)
        return round(max(0.0, 0.3 * (1.0 - min(1.0, distance))), 4)


ResumeScreeningEnvironment = ResumeEnv
