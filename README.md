# dqn-cart-pole

Double DQN agent that learns Gymnasium's CartPole on CPU.

## Setup

Requires [uv](https://docs.astral.sh/uv/), which also installs a managed Python matching `requires-python`.

```sh
uv sync
prek install
```

## Train

```sh
uv run dqn train
uv run dqn train --total-steps 50000 --seed 2
```

Each run gets its own folder under `runs/`:

| file            | contents                                  |
| --------------- | ----------------------------------------- |
| `config.json`   | hyperparameters used                      |
| `metrics.jsonl` | episode, learn and evaluation records     |
| `training.log`  | lifecycle log                             |
| `model.pt`      | weights from the best evaluation          |
| `episode.gif`   | the best agent playing one greedy episode |

## Development

```sh
uv run pytest
prek run --all-files

scripts/clean.sh        # caches only
scripts/clean.sh --all  # also runs, .venv and build metadata
```

On Windows: `scripts/clean.ps1` and `scripts/clean.ps1 -All`, with `-WhatIf` for a dry run.
