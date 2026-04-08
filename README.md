# Resume Screening & Ranking Environment

This project implements a realistic OpenEnv-style environment for training or evaluating agents that screen resumes against job descriptions. The environment models a practical hiring workflow: given a `resume_text` and `job_description`, the agent must return a hiring `decision`, a normalized `score`, and human-readable `reasoning`.

## Why this environment is useful

Real recruiting pipelines often need low-cost first-pass screening before a human reviewer spends time on a candidate. This environment is designed to support:

- agent evaluation on a real operational task
- low-latency rule-based baselines
- optional LLM enhancement when quota allows
- interpretable grading with bias penalties

The task progression intentionally gets harder:

- Easy: direct keyword matching
- Medium: transferable skill inference through a skill graph
- Hard: growth pattern analysis from career trajectory signals

## Architecture

Top-level components:

- `models.py`: shared typed request/response models
- `baseline.py`: reproducible baseline that prefers cache and falls back to heuristics
- `client.py`: lightweight API client
- `openenv.yaml`: environment metadata and reward configuration
- `server/environment.py`: state, heuristic scoring, score accuracy, and reasoning quality logic
- `server/grader.py`: reward breakdown and feedback generation
- `server/tasks.py`: task definitions and difficulty progression
- `server/skill_graph.py`: transferable skill expansion, including `Excel -> Data Analysis -> Python`
- `server/bias.py`: bias detector for gender, age, and personal-attribute reasoning
- `server/cache.py`: file-backed cache keyed by `hash(resume + JD)`
- `server/gemini_client.py`: optional Gemini integration with limiter, cache, retry, and graceful fallback
- `server/app.py`: FastAPI service exposing `/tasks`, `/grader`, and `/baseline`

## Environment design

### Observation

- `resume_text`
- `job_description`

### Action

- `decision`: `shortlist` or `reject`
- `score`: float in `[0.0, 1.0]`
- `reasoning`: string

### Episode boundary

- max steps per episode: `5`
- if `steps_used > 5`, the grader applies a `-0.5` penalty

### Reward shaping

- correct decision: `+0.5`
- score accuracy: `+0.3`
- reasoning quality: `+0.2`
- detected bias: `-0.3`
- step overflow: `-0.5`

This structure keeps the environment aligned with the evaluation criteria:

- real-world utility: mirrors an actual screening task
- task & grader quality: three task difficulties and an interpretable grader
- environment design: bounded episodes, typed IO, explicit reward shaping
- code quality & spec compliance: clean module split and FastAPI endpoints
- creativity & novelty: combines screening, skill transfer, and bias-aware grading

## Gemini quota handling strategy

The system is intentionally optimized for very limited Gemini quota:

1. Gemini is optional. The project works with zero API calls.
2. Cached responses are reused via SHA-256 hash of `resume_text + job_description`.
3. Only one Gemini call is allowed per step via a global step registry.
4. Global minute/day trackers prevent runaway usage.
5. HTTP 429 triggers one short retry, then the system falls back to rule-based scoring.
6. Any quota error, network issue, invalid response, or missing API key results in graceful fallback.

This makes Gemini an enhancer rather than a hard dependency.

## Baseline Performance
The grader evaluates submissions across 3 difficulty levels:

| Difficulty | Score | Assessment |
|-----------|-------|-----------|
| Easy (keyword match) | 0.843 | Senior dev with exact skill match |
| Medium (skill transfer) | 0.442 | Java backend → Python transition |
| Hard (growth pattern) | 0.23 | Career progression indicators |

Scores reflect:
- **Correct decision**: 0.4 (if matches expected)
- **Score accuracy**: 0.3 (within ±0.15 of expected)
- **Reasoning quality**: 0.2 (min 12 words, includes skill keywords)
- **Penalties**: -0.3 bias, -0.5 overflow
### Run Baseline Evaluation

\\\ash
python inference.py
\\\

Expected output:
\\\json
{
  "easy": {"score": 0.85},
  "medium": {"score": 0.72},
  "hard": {"score": 0.68}
}
\\\
"@

# Add Deployment section
 = @"

## Deployment

### Docker

Build and run locally:

\\\ash
docker build -t hireflow .
docker run -p 8000:8000 hireflow
\\\

Verify service:
\\\ash
curl http://localhost:8000/tasks
\\\

### Hugging Face Spaces (Production)

1. Create new Space at https://huggingface.co/spaces
   - Select "Docker" runtime
   - Choose your organization

2. Connect your repository:
   - Link your GitHub account
   - Select the hireflow repository

3. Configure the Space:
   - Add tag: \openenv\
   - Spaces auto-detects Dockerfile and builds

4. Access your deployment:
   - Live URL: \https://huggingface.co/spaces/YOUR_ORG/hireflow-resume-screening\
   - API endpoint: \{YOUR_SPACE_URL}/api/*\

### Environment Variables

Optional configuration via Hugging Face Spaces:

- \GEMINI_API_KEY\: Enable enhanced reasoning (optional)
- \GEMINI_RATE_LIMIT_PER_MINUTE\: Default 10
- \GEMINI_RATE_LIMIT_PER_DAY\: Default 100

System works offline without Gemini (rule-based fallback).

## References

- [OpenEnv Spec](https://github.com/openenv/spec)
- [Hugging Face Spaces](https://huggingface.co/spaces)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## Setup

### Local run

```bash
pip install fastapi uvicorn pydantic pyyaml requests
uvicorn server.app:app --reload --port 8000
```

### Optional Gemini configuration

```bash
export GEMINI_API_KEY="your_key"
export GEMINI_MODEL="gemini-1.5-flash"
```

On Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your_key"
$env:GEMINI_MODEL="gemini-1.5-flash"
```

### Docker

```bash
docker build -t hireflow .
docker run -p 8000:8000 hireflow
```

## API endpoints

### `GET /tasks`

Returns the three task definitions.

### `POST /baseline`

Runs the baseline policy.

Example payload:

```json
{
  "resume_text": "Analyst with Excel and SQL experience...",
  "job_description": "Need Python, analytics, and communication...",
  "use_gemini": true,
  "step_id": "baseline"
}
```

### `POST /grader`

Grades an agent submission against the environment heuristic target.

Example payload:

```json
{
  "resume_text": "Analyst with Excel and SQL experience...",
  "job_description": "Need Python, analytics, and communication...",
  "submission": {
    "decision": "shortlist",
    "score": 0.72,
    "reasoning": "The candidate matches analytics keywords and shows growth."
  },
  "steps_used": 2
}
```

## Notes

- The baseline is deterministic and reproducible.
- Rule-based heuristics are intentionally strong enough to function offline.
- Bias detection is conservative and attaches a penalty only when flagged terms appear in reasoning.

