# dqn-cart-pole

Double DQN agent that learns Gymnasium's CartPole on CPU.

## Requirements

[uv](https://docs.astral.sh/uv/), which also installs the Python it needs:

```sh
winget install astral-sh.uv
```

PyTorch has no wheels for Python 3.15 yet, so the environment has to be built on 3.14:

```sh
uv python install 3.14
uv sync --python 3.14
```

## Train

```sh
uv run src/main.py train
uv run src/main.py train --total-steps 50000 --seed 2
```

Each run gets its own folder under `runs/`:

| file            | contents                                  |
| --------------- | ----------------------------------------- |
| `config.json`   | hyperparameters used                      |
| `metrics.jsonl` | episode, learn and evaluation records     |
| `training.log`  | lifecycle log                             |
| `model.pt`      | weights from the best evaluation          |
| `episode.gif`   | the best agent playing one greedy episode |
