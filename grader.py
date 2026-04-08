from __future__ import annotations

from models import GraderRequest, GraderResponse, RewardBreakdown
from server.bias import detect_bias
from server.environment import ResumeEnv


def grade_score(resume_text: str, job_description: str, submission_decision: str, submission_score: float) -> float:
    """
    Deterministic grader that returns a single score between 0.0 and 1.0.
    
    Scoring breakdown:
    - Decision correctness: 0.4 (0.4 if correct, 0.0 otherwise)
    - Score alignment: 0.3 (based on distance from expected)
    - Reasoning quality: 0.2 (implicit, evaluated through decision/score match)
    
    Args:
        resume_text: Resume content
        job_description: Job description
        submission_decision: Agent's decision ("shortlist" or "reject")
        submission_score: Agent's score (0.0-1.0)
    
    Returns:
        Score between 0.0 and 1.0
    """
    environment = ResumeEnv()
    expected_output, _ = environment.build_rule_based_output(resume_text, job_description)
    
    # Component 1: Decision correctness (0.4 weight)
    decision_score = 0.4 if submission_decision == expected_output.decision else 0.0
    
    # Component 2: Score alignment (0.3 weight)
    score_distance = abs(expected_output.score - submission_score)
    score_component = 0.3 * max(0.0, 1.0 - score_distance)
    
    # Component 3: Bonus for good combination (0.2 weight from reasoning quality)
    reasoning_bonus = 0.2 if (
        submission_decision == expected_output.decision and 
        abs(expected_output.score - submission_score) < 0.15
    ) else 0.0
    
    # Penalty for extreme misalignment
    penalty = 0.0 if (score_distance < 0.5 or submission_decision == expected_output.decision) else 0.1
    
    total_score = max(0.0, min(1.0, decision_score + score_component + reasoning_bonus - penalty))
    return round(total_score, 4)


def grade_submission(request: GraderRequest) -> GraderResponse:
    """
    Full grading response with detailed breakdown and feedback.
    """
    environment = ResumeEnv()
    expected_output, heuristic = environment.build_rule_based_output(request.resume_text, request.job_description)
    bias = detect_bias(request.submission.reasoning)

    # Deterministic scoring components
    correct_decision = 0.4 if request.submission.decision == expected_output.decision else 0.0
    score_accuracy = 0.3 * max(0.0, 1.0 - abs(expected_output.score - request.submission.score))
    reasoning_quality = environment.reasoning_quality(
        request.submission.reasoning, request.resume_text, request.job_description
    )
    
    # Bias and step penalties
    bias_penalty = bias.penalty
    overflow_penalty = 0.5 if request.steps_used > environment.max_steps else 0.0

    # Total reward (clamped to 0.0-1.0)
    total_reward = round(
        max(0.0, min(1.0, correct_decision + score_accuracy + reasoning_quality - bias_penalty - overflow_penalty)),
        4,
    )

    reward = RewardBreakdown(
        correct_decision=round(correct_decision, 4),
        score_accuracy=round(score_accuracy, 4),
        reasoning_quality=round(min(0.2, reasoning_quality), 4),
        bias_penalty=round(bias_penalty, 4),
        overflow_penalty=round(overflow_penalty, 4),
        total_reward=total_reward,
    )

    metrics = {
        "expected_score": expected_output.score,
        "keyword_score": heuristic.keyword_score,
        "transferable_score": heuristic.transferable_score,
        "growth_score": heuristic.growth_score,
        "steps_used": float(request.steps_used),
        "final_grade": total_reward,
    }

    # Enhanced feedback
    decision_feedback = "✓" if request.submission.decision == expected_output.decision else "✗"
    score_feedback = f"±{abs(expected_output.score - request.submission.score):.3f}"
    
    feedback = (
        f"[Decision {decision_feedback}] Expected '{expected_output.decision}' vs '{request.submission.decision}'. "
        f"[Score {score_feedback}] Expected {expected_output.score:.2f} vs {request.submission.score:.2f}. "
        f"Reasoning score: {reward.reasoning_quality:.2f}/0.20. "
        f"Final grade: {total_reward:.2f}/1.00"
    )

    return GraderResponse(
        reward=reward,
        expected_output=expected_output,
        bias=bias,
        metrics=metrics,
        feedback=feedback,
    )
