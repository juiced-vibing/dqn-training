# AGENTS.md

Instructions for coding agents working in this repo. Humans: see [README.md](README.md).

dqn-cart-pole trains a Double DQN agent on Gymnasium's CartPole, on CPU, and saves the best weights, the metrics and a gif of the agent playing.

## Layout

`src/` is the import root, not a package: modules import each other as `from agent.dqn_agent import DQNAgent`.

| path                    | contents                                                                                |
| ----------------------- | --------------------------------------------------------------------------------------- |
| `src/main.py`           | entry point: `uv run src/main.py train`                                                 |
| `src/cli.py`            | the typer commands; creates the run folder and writes its files                         |
| `src/config.py`         | `TrainingConfig`: every hyperparameter, validated once when it is built                 |
| `src/training.py`       | the training loop                                                                       |
| `src/agent/`            | the agent, its Q-network and the replay buffer                                          |
| `src/env/spec.py`       | `EnvSpec`: observation and action sizes, or the error for an env we cannot train on     |
| `src/env/evaluation.py` | greedy evaluation on held-out seeds, and the recorded episode                           |
| `src/utils/`            | metrics records and their JSONL writer, the rich display, logging, the epsilon schedule |
| `tests/`                | pytest, mostly edge cases and probe problems with a known answer                        |

## Commands

```sh
uv python install 3.14   # torch has no wheels for 3.15 yet
uv sync --python 3.14
uv run src/main.py train --total-steps 50000

uv run pytest
uv run ruff check --fix
uv run ruff format
uv run ty check

prek install           # once
prek run --all-files   # everything above, plus ssort and pyproject-fmt
```

## Conventions

- Make invalid states unrepresentable instead of checking for them: a value is validated once by its type (`TrainingConfig`, `EnvSpec`), and code that receives one does not check again. Metrics are a tagged union of frozen structs, handled with `match`.
- Errors this package raises on purpose derive from `DQNError` in `src/errors.py`.
- No docstrings, and few comments: a short lower-case `#` line only where the reason is not obvious. The two docstrings in `cli.py` stay, they are the `--help` text.
- Tests are for edge cases and invariants, not for restating the code.
- Split a file when it holds two responsibilities; around 250 lines is the point to check.
- Runs, weights and gifs are never committed (`runs/`, `*.pt`, `*.gif`).

## Commits

Conventional commits, lower case, one short line, no body unless it is needed: `feat: …`, `fix(agent): …`, `refactor: …`, `chore: …`, `docs: …`. No co-author trailers.
