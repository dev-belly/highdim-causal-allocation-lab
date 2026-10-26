# Contributing to High-Dimensional Causal Allocation Lab

Monte Carlo experiments on high-dimensional covariate adjustment and downstream portfolio estimation risk, with a browser simulation lab.

Bug reports, documentation fixes and focused improvements are welcome. English and Chinese are both fine. For setup and questions, start with [SUPPORT.md](SUPPORT.md) and the [README](README.md). Follow [SECURITY.md](SECURITY.md) for suspected vulnerabilities.

## Prepare a change

Check existing issues and pull requests to avoid duplicating work. For a larger design or research change, describe the problem, assumptions and acceptance criteria in an issue first. Keep each pull request focused and explain the resulting behavior for a reader who has not followed the discussion.

Use Node.js 22.12+ and Python 3.10/3.12 as exercised by CI.

```bash
npm ci
# Python checks below use the locked environment in python/.
```

Activate the virtual environment with the command appropriate for your shell. Run commands from the repository root unless a command changes directory.

## Validate

The [CI workflow](.github/workflows/ci.yml) is the current reference for automated checks. Run the checks relevant to your change and report the actual outcome:

```bash
npm test
npm run build
cd python
uv sync --locked --extra dev --python 3.12
uv run --no-sync ruff check src tests
uv run --no-sync pytest tests -q
```

Add or update a meaningful regression test when behavior changes. A documentation-only change needs review of its commands, links and claims; do not invent test outcomes or benchmark results. If a check was skipped, say so in the pull request.

## Preserve evidence

Include the frontend or Python backend, estimator, n/p grid, DGP parameters, seed, repetitions and the affected metric. Distinguish Monte Carlo uncertainty, confidence-interval coverage and portfolio utility; synthetic experiments do not establish a real-market causal effect.

Report the browser version for UI or worker problems, and the Python version for simulation problems. State which implementation produced the result. A seed and small DGP configuration usually suffice; do not upload proprietary observations or personal data.

Keep historical results distinguishable from reruns. Record configurations, seeds, environment and source revision when adding experiments. Avoid choosing favorable seeds or changing benchmark definitions to improve a headline number. Describe limitations and label proposed features as planned until implemented and verified.

## Submit

Use a descriptive branch and commit message, fill in the pull request template, and link any related issue. Include concise validation evidence and update relevant documentation. Do not include generated local caches, credentials, private datasets or unrelated edits. Contributions are reviewed as capacity permits; no response or merge date is promised.
