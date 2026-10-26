# Support

Monte Carlo experiments on high-dimensional covariate adjustment and downstream portfolio estimation risk, with a browser simulation lab.

## Start here

Read the [README](README.md) for setup, examples and method boundaries. The tested maintenance commands are defined in the [CI workflow](.github/workflows/ci.yml). Environment: Node.js 22.12+ and Python 3.10/3.12 as exercised by CI.

```bash
npm ci
# Python checks below use the locked environment in python/.
```

Report the browser version for UI or worker problems, and the Python version for simulation problems. State which implementation produced the result. A seed and small DGP configuration usually suffice; do not upload proprietary observations or personal data.

## Ask a question or report a problem

Use [GitHub Issues](https://github.com/dev-belly/highdim-causal-allocation-lab/issues) for usage questions, reproducible bugs and scoped feature proposals. English and Chinese reports are welcome. Check existing issues first and use the bug or feature form when it fits. Support is best effort; no response time or delivery date is promised.

Include the frontend or Python backend, estimator, n/p grid, DGP parameters, seed, repetitions and the affected metric. Distinguish Monte Carlo uncertainty, confidence-interval coverage and portfolio utility; synthetic experiments do not establish a real-market causal effect.

Include the commit SHA, operating system, dependency versions, exact command, expected behavior and a minimal reproduction. Share only redacted logs and data you are allowed to publish. Remove credentials, personal information and confidential or licensed records. For a suspected security vulnerability, follow [SECURITY.md](SECURITY.md) before posting details.

## Verify locally

Run relevant checks before reporting a regression or opening a pull request:

```bash
npm test
npm run build
cd python
uv sync --locked --extra dev --python 3.12
uv run --no-sync ruff check src tests
uv run --no-sync pytest tests -q
```

These are verification commands, not a claim that a new environment has already passed them. For changes that affect results, retain input provenance, configuration and seeds alongside the output.
