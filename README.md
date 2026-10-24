<div align="center">
    <img src="https://github.com/MyNameIsArko/RL-Flax/assets/47687092/6886a474-8e6a-4953-9b81-3f67b6a8da68" alt="logo">
</div>

# RL-Flax
This is an attempt to recreate many reinforcement learning algorithms in Jax(Flax) world as single-file implementations.
## Run Locally

Clone the project

```bash
  git clone https://github.com/MyNameIsArko/RL-Flax
```

Go to the project directory

```bash
  cd RL-Flax
```

## Install

```bash
uv sync --locked
```

This project uses standard CPython 3.14 and uv. Commands run in the project-local `.venv`. The default setup uses CPU JAX.

Install every optional feature with `uv sync --locked --all-extras`, or select only what a script needs:

- `mujoco` for continuous-action environments such as Pusher.
- `atari` for ALE environments and OpenCV preprocessing.
- `video` for video recording.
- `tracking` for W&B and TensorBoard.

## Run

```bash
uv run --locked python PPO.py --help
uv run --locked --extra mujoco python PPO_continuous_action.py --help
uv run --locked --extra atari python PPO_atari.py --help
```

Run the synthetic checks with:

```bash
uv run --locked --all-extras python -B -W error::DeprecationWarning -W error::FutureWarning -m unittest discover -s tests -p 'test_modernization.py' -v
```

CUDA setup and learning performance are not covered by these instructions.


## Roadmap

- DQN

- Rainbow DQN

- A2C

- A3C


## Contributing

Any kind of contribution is welcome!

If you know a little bit of Jax+Flax and know ins and outs of some algorithm then make a pull request. I'll gladly accept it as this is a big project for one man.
## Acknowledgements
 - [PPO implementation details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
 - [CleanRL](https://github.com/vwxyzjn/cleanrl)
 - [Jax documentation](https://jax.readthedocs.io/en/latest/)
 - [Flax documentation](https://flax.readthedocs.io/en/latest/)


## License

[MIT](https://choosealicense.com/licenses/mit/)

