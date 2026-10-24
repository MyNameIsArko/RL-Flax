# Modernization plan: uv, Python 3.14, NNX, and Gymnasium

Start with Phase 0 to establish the Python environment. Complete each phase before moving to the next phase.

This is an implementation specification, not a record of completed code changes. The review date is October 24, 2026. The three training scripts remain unchanged. The plan targets Python 3.14 and current Flax NNX, JAX, Gymnasium, and Orbax interfaces.

## Scope and limits

Implement these changes after the uv setup in Phase 0:

1. Replace Linen models and `TrainState` with native NNX models and an NNX optimizer.
2. Replace removed interfaces and avoid deprecated interfaces in the code that this migration touches.
3. Update Gymnasium environments, wrappers, episode handling, and statistics.
4. Preserve PPO behavior except for the correctness fixes specified below.
5. Add focused tests that use synthetic data and small local environments.

Keep the three standalone implementations. Do not introduce a shared training framework or a cross-file runtime helper. Some repeated code is intentional because each script must remain readable on its own.

Create a uv project with `pyproject.toml`, `.python-version`, and `uv.lock`. Install its dependencies into the local `.venv`. Update the README's installation and execution instructions. Do not add a separate requirements file because `pyproject.toml` and `uv.lock` are the dependency sources.

Package and Python downloads are authorized during implementation. Do not contact W&B, run training, or add algorithms, device selection, parallel training, or performance experiments. Current ALE packages include their ROMs, so do not add AutoROM or a separate ROM downloader.

Use the newest documented interfaces and test them in the uv environment. Report resolver and installation failures with the exact package, platform, and error. Do not silently downgrade Python or add an old-version compatibility layer.

Estimate 13–21 implementation hours for setup, code, and bounded tests. Allow 5–20 minutes for downloads on a working broadband connection. These estimates exclude learning experiments. A smaller model can use a separate implementation turn for each phase.

## Evidence and API decisions

### Read the distinction between migration and deprecation

Linen is still supported. Its replacement here is an explicit project decision. `flax.struct.dataclass` is also supported and can remain for array storage. `flax.training.orbax_utils` still exists, so do not describe it as removed.

`jax.random.PRNGKeyArray` is removed and must disappear. `jax.random.PRNGKey` is a legacy key constructor, so replace it with `jax.random.key`. Use `jax.Array` for JAX array annotations. Do not claim that every legacy interface is formally deprecated. [JAX changelog](https://docs.jax.dev/en/latest/changelog.html), [Flax migration guide](https://flax.readthedocs.io/en/latest/migrating/linen_to_nnx.html)

### Follow the current NNX contracts

Use these contracts rather than older NNX examples:

1. Import `nnx` with `from flax import nnx`. Define layers in `__init__`, then call the model directly.
2. Construct `nnx.Linear(in_features, out_features, rngs=rngs)`. NNX needs both dimensions at construction time.
3. Construct `nnx.Optimizer(agent, tx, wrt=nnx.Param)`. The `wrt` argument is required.
4. Apply gradients with `optimizer.update(agent, grads)`. Do not use the old one-argument update call.
5. Read array-backed variables with `variable[...]`. Avoid `.value` and `.raw_value` access.

Use `@nnx.jit` for functions that accept NNX modules or optimizers. Use `nnx.value_and_grad(..., has_aux=True)` for the PPO gradient. Use plain `@jax.jit` only for functions that receive arrays and scalars. Do not use experimental graph modes or a Linen bridge. [Optimizer reference](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/training/optimizer.html), [transformation reference](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html)

Use `nnx.Sequential(layer, activation, layer)` with positional arguments. Do not pass the old Linen list as one argument. Use `nnx.Param` for the learned continuous-action log standard deviation. Use the existing initializer scales through `nnx.initializers`. [Layer reference](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html), [sequential reference](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/helpers.html), [initializer reference](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/initializers.html)

The current source explicitly marks `.value` and `.raw_value` access as deprecated. Some examples elsewhere still use them. Prefer array indexing in the new code. [Flax variable source](https://flax.readthedocs.io/en/latest/_modules/flax/nnx/variablelib.html)

### Fix Gymnasium against one explicit reset mode

Use `gym.vector.SyncVectorEnv(..., autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)` in every script. A same-step reset returns the next episode's initial observation when the previous episode ends. It also exposes the final observation through `infos["final_obs"]`.

Current `final_info` is a dictionary of batched fields. It is not the old list of dictionaries. Use the masks for ended episodes. Do not support several old formats through exception handling. [Gymnasium autoreset reference](https://gymnasium.farama.org/api/vector/autoreset/)

### Use current checkpoint interfaces

Use `from orbax.checkpoint import v1 as ocp`. This is the public alias exported by the current Orbax package. Use blocking `ocp.save(path, state, custom_metadata=metadata)` for the single final save. Use `ocp.load(path, abstract_state=target)` in restoration tests.

Do not copy the older `PyTreeCheckpointer` and `save_args_from_target` sequence. Do not import the experimental namespace directly. Do not add a checkpoint manager for a single save. [Orbax V1 reference](https://orbax.readthedocs.io/en/latest/api_reference/checkpoint.v1.html), [public namespace source](https://github.com/google/orbax/blob/main/checkpoint/orbax/checkpoint/__init__.py)

## File map

The line numbers below describe the original checkout. Use function and class names after edits because line numbers will move.

| File | Original locations | Required work |
| --- | --- | --- |
| `PPO.py` | Models: 30–65. Rollout and GAE: 139–222. Updates: 225–309. Startup and save: 312–420. | Establish the complete discrete reference implementation. |
| `PPO_continuous_action.py` | Models: 30–66. Wrappers: 101–111. Probability code: 142–171. Startup and save: 316–424. | Port common fixes, then preserve Gaussian actions and continuous wrappers. |
| `PPO_atari.py` | Models: 30–73. Environment: 120–130. Updates: 244–327. Startup and save: 330–446. | Port common fixes, then preserve the shared convolutional network. |
| `tests/test_modernization.py` | New file. | Add standard-library `unittest` tests, synthetic fixtures, and static acceptance tests. |
| `MODERNIZATION_PLAN.md` | This file. | Use it as the handoff specification. Update completion notes only if asked. |

The project setup also changes these files:

| File | Required work |
| --- | --- |
| `pyproject.toml` | Declare the uv application, Python support, core dependencies, and feature extras. |
| `.python-version` | Select standard CPython 3.14 for project commands. |
| `uv.lock` | Record the resolved package versions and platform-specific distributions. |
| `.gitignore` | Allow the Python version file and lockfile in version control. Keep `.venv` ignored. |
| `README.md` | Replace the old installation instructions with tested uv commands and feature descriptions. |

## Phase 0: Create and install the uv project

Estimate: 30–60 minutes, plus download time. Do this before editing the algorithms. uv is already installed on the review machine. Do not reinstall or update a working uv installation without a specific need.

### 0.1 Initialize the existing repository without restructuring it

Use standard CPython 3.14, with its normal interpreter lock. Do not select the separate free-threaded `3.14t` build. Use a project-local environment at `.venv`, not a global Python environment.

Run these commands from the repository root, in this order:

```bash
uv python install 3.14
uv init --bare --name rl-flax --python 3.14 --no-workspace
uv python pin 3.14
```

Before initialization, make sure that `pyproject.toml` does not already exist. If it exists, inspect and extend it instead of overwriting it or rerunning `uv init`. Do not create a parent uv workspace or modify a parent project's files.

Set `requires-python = ">=3.14"` in `[project]`. Keep `.python-version` at `3.14` to select the latest installed patch release in that minor series. Record the exact installed patch release in the final report. Python 3.15 is not an acceptance target for this pass.

Add this configuration to `pyproject.toml`:

```toml
[tool.uv]
package = false
```

This project runs standalone files and does not need a build backend, a `src/` tree, package discovery, or console entry points. Do not create a sample `main.py` or overwrite the existing README. [uv initialization reference](https://docs.astral.sh/uv/concepts/projects/init/), [Python version selection](https://docs.astral.sh/uv/concepts/python-versions/)

The existing `.gitignore` ignores `.python-version`. Remove that one ignore rule so the version file can enter version control. Preserve unrelated rules. Keep `.venv` ignored, and do not ignore `uv.lock`.

### 0.2 Declare dependencies by feature

Use `uv add` to record current stable dependencies in `pyproject.toml`. Let uv add lower bounds from the selected releases. The lockfile records the exact resolution. Do not copy arbitrary old version numbers or select prereleases to bypass a failure.

The core environment must support discrete PPO, numerical tests, and checkpointing:

```bash
uv add jax flax optax numpy "gymnasium[classic-control]" \
    orbax-checkpoint tensorboardx tqdm
```

Declare these optional extras separately:

```bash
uv add --optional mujoco "gymnasium[mujoco]"
uv add --optional atari "gymnasium[atari]" opencv-python
uv add --optional video moviepy
uv add --optional tracking wandb tensorboard
```

`mujoco` supports the continuous script's default Pusher environment. `atari` includes ALE and OpenCV for preprocessing. `video` supports `RecordVideo`. `tracking` includes TensorBoard for W&B synchronization as well as W&B itself.

Current Gymnasium's `classic-control` extra selects `pygame-ce`, which provides Python 3.14 distributions. Do not add the older `pygame` distribution alongside it. Both distributions provide the same import namespace. [Gymnasium metadata](https://pypi.org/pypi/gymnasium/json), [pygame-ce distributions](https://pypi.org/project/pygame-ce/)

Do not add `tensorflow`, `tensorflow-probability`, Distrax, PyTorch, pytest, or a separate `jaxlib` pin. JAX selects its compatible compiled library. Keep standard-library `unittest` for tests. Do not add a development dependency group when no extra test package is needed.

Select regular CPU JAX for this Mac. Do not use the README's old CUDA installation command. Do not add CUDA extras to the universal default environment. GPU installation and performance remain outside this pass. [JAX installation reference](https://docs.jax.dev/en/latest/installation.html)

Package metadata reviewed on October 24, 2026 includes JAX 0.12.0.1, Flax 0.12.10, and Gymnasium 1.4.0. These are reference versions for the API review, not a claim that the full set passed runtime tests. Select current stable releases during implementation and record the actual resolved set. [JAX package](https://pypi.org/project/jax/), [Flax package](https://pypi.org/project/flax/), [Gymnasium package](https://pypi.org/project/gymnasium/)

### 0.3 Resolve and install both the core and full environments

Generate `uv.lock`, then establish the core environment:

```bash
uv lock
uv sync --locked
uv run --locked python -c "import sys; import jax; from flax import nnx; import optax; import gymnasium; from orbax.checkpoint import v1; print(sys.version); print(jax.devices())"
```

The library import command does not import the old PPO scripts. Those scripts still contain known broken imports until their migration finishes. Do not add TensorFlow Probability to make them import during Phase 0.

Install all declared features into the same environment:

```bash
uv sync --locked --all-extras
uv run --locked --all-extras python -c "import ale_py, mujoco, cv2, moviepy, wandb, tensorboard; print('Feature imports passed')"
uv tree --locked
```

Use `--all-extras` for subsequent full-suite commands. A core-only synchronization can remove optional packages from `.venv`. Do not interpret that removal as a project dependency regression. Use the documented sync mode consistently. [uv project commands](https://docs.astral.sh/uv/guides/projects/), [sync behavior](https://docs.astral.sh/uv/reference/cli/)

Make sure that the project interpreter reports version 3.14 and lives in this repository's `.venv`. Make sure that `jax.devices()` reports a usable CPU device. Inspect the installed NNX optimizer and Orbax V1 interfaces before starting the code migration.

If installation fails, inspect the exact requirement and available Python 3.14 distributions. Prefer a released compatible version that still satisfies the current API contracts. Do not switch to Python 3.13, nightly packages, Rosetta, or native compilation without reporting why the selected target failed.

Do not claim success from package metadata or dependency resolution alone. The installed library imports must actually execute. Record each failed feature group separately and keep successful core setup intact.

### 0.4 Make setup reproducible in the README

Replace the old `pip install "jax[cuda]"` and TensorFlow Probability instructions. Explain that the project uses Python 3.14, uv, and a local `.venv`. Preserve unrelated project text and attribution.

Document these common commands:

```bash
uv sync --locked
uv sync --locked --all-extras
uv run --locked python PPO.py --help
uv run --locked --extra mujoco python PPO_continuous_action.py --help
uv run --locked --extra atari python PPO_atari.py --help
```

Explain each extra and the default CPU installation. Show the full synthetic-test command from Phase 5 after those tests exist. Do not claim Python 3.15, CUDA, or learning performance support from this setup.

Do not run training commands as a setup test. The help commands become valid acceptance checks after Phases 1–4. Importing W&B for a dependency test must not start a W&B run or request login.

### 0.5 Define the setup exit conditions

Make sure that these conditions hold:

1. `pyproject.toml` declares Python 3.14, all direct core imports, and the four feature extras.
2. `.python-version` selects 3.14 and can enter version control.
3. `uv.lock` exists and `uv sync --locked --all-extras` succeeds without rewriting it.
4. Core and feature imports execute in the project environment, with their versions recorded.
5. No training ran and no unrelated source layout or parent workspace changed.

Keep the three project files under version control with the implementation changes. Do not commit `.venv`, caches, logs, videos, downloaded Python runtimes, or generated model files. Do not generate a duplicate requirements file.

## Phase 1: Make discrete PPO a native NNX implementation

Estimate: 3–5 hours. Edit `PPO.py` first. Do not port an incomplete pattern to the other scripts.

### 1.1 Make functions importable without starting a run

Complete these actions:

1. Put the executable setup and loop in `main()`.
2. Keep only `main()` under `if __name__ == "__main__":`.
3. Change `parse_args()` to `parse_args(argv=None)` and pass `argv` to `parser.parse_args`.
4. Remove runtime dependence on module-level `args` and `envs` from helper functions.
5. Pass dimensions, arrays, and needed scalar values explicitly to numerical helpers.

Importing the module must not construct an environment, open a writer, register a signal handler, or create files. Signal handling belongs in `main`. Keep W&B imports inside the enabled tracking path.

Pass `args` and `envs` to ordinary Python orchestration functions where needed. Do not pass an `argparse.Namespace`, a Gymnasium environment, or a writer into a compiled function. Pass individual coefficients to compiled numerical functions instead.

### 1.2 Replace the models and state containers

Delete `AgentParams` and `AgentState`. Remove `FrozenDict`, `TrainState`, callback fields, and their unused typing imports. Keep the supported Flax dataclass for `Storage`. Replace `jnp.array` type annotations with `jax.Array`.

Define these NNX objects:

1. `Actor(obs_dim, action_n, *, rngs)` owns layers with dimensions `obs_dim -> 64 -> 64 -> action_n`.
2. `Critic(obs_dim, *, rngs)` owns layers with dimensions `obs_dim -> 64 -> 64 -> 1`.
3. `Agent(obs_dim, action_n, *, rngs)` owns one actor and one critic.

Keep `tanh` after each hidden layer. Keep orthogonal scales of `sqrt(2)` for hidden layers, `0.01` for the actor output, and `1.0` for the critic output. Keep zero biases.

Change `linear_layer_init` to accept `in_features`, `out_features`, and keyword-only `rngs`. It must return an `nnx.Linear`. Create the layers once during construction. Do not create layers inside `__call__`.

Use one `nnx.Rngs` object during construction so successive layers receive different keys. Do not recreate `nnx.Rngs` with the same seed for each layer. Keep action sampling and minibatch shuffling on separate explicit JAX keys.

Split `jax.random.key(args.seed)` into three keys for model initialization, action sampling, and minibatch shuffling. Pass the initialization key to `nnx.Rngs`. Advance and return each runtime key from its own function. Remove unused root keys and the old separate actor/critic initialization calls.

Define `Agent.__call__(obs)` to return `(logits, values)`. Require `obs.shape == (batch, obs_dim)`, `logits.shape == (batch, action_n)`, and `values.shape == (batch,)`. Remove only the final critic dimension with `jnp.squeeze(value, axis=-1)`. An unrestricted `.squeeze()` incorrectly removes a batch dimension when the batch contains one item.

Require a one-dimensional `Box` observation space for this discrete script. Derive `obs_dim` from `envs.single_observation_space.shape[0]`. Reject unsupported observation layouts with a clear error instead of silently changing network behavior.

### 1.3 Replace TensorFlow Probability

Remove the TensorFlow Probability import. Do not add Distrax because these scripts need only two simple distributions.

Use these categorical operations:

```python
actions = jax.random.categorical(sample_key, logits, axis=-1).astype(jnp.int32)
log_probs = jax.nn.log_softmax(logits, axis=-1)
selected_log_probs = jnp.take_along_axis(
    log_probs, actions[..., None], axis=-1
)[..., 0]
entropy = -(jnp.exp(log_probs) * log_probs).sum(axis=-1)
```

The action, selected log probability, entropy, and value arrays must each have shape `(batch,)`. Allocate discrete action storage with `dtype=jnp.int32`. Do not store category indices in floating-point arrays. Use the finite log probabilities from `log_softmax` directly, even when their exponentials underflow to zero. Do not use `entr(exp(log_probs))` because its gradients become nonfinite at zero probability. [Categorical sampling](https://docs.jax.dev/en/latest/_autosummary/jax.random.categorical.html), [log-softmax](https://docs.jax.dev/en/latest/_autosummary/jax.nn.log_softmax.html)

Keep `get_action_and_value` as the sampled-action function. Give it `(agent, obs, key)` and return `(actions, logprobs, values, next_key)`. Split the key once, sample with the subkey, and return the unused parent branch. This function does not receive or mutate storage.

Keep `get_action_and_value2` as the existing-action evaluation function. Give it `(agent, obs, actions)` and return `(logprobs, entropy, values)`. It does not sample or accept a key. Use the same categorical probability calculation in both paths.

Move buffer writes into `rollout`. This keeps environment control outside NNX transforms and makes both probability functions independently testable.

### 1.4 Replace the optimizer and update step

Construct one optimizer over the entire `Agent`. Preserve the current `optax.chain`, global gradient clipping, AdamW algorithm, epsilon `1e-5`, default weight decay, and learning-rate decay. This pass does not tune PPO.

Remove `optax.inject_hyperparams` if its only remaining purpose is reading the learning rate. Supply the schedule directly to `optax.adamw`. Preserve the original schedule's equation:

```python
updates_per_rollout = num_minibatches * update_epochs
rollout_index = count // updates_per_rollout
learning_rate = initial_learning_rate * (1.0 - rollout_index / num_updates)
```

Create the schedule once after parsing arguments. Capture only immutable scalar values. Do not read a global namespace from the schedule.

Change `ppo_loss` to receive the agent as its first argument. Remove its separate parameter-tree argument. Keep the current policy clipping, advantage normalization, clipped value loss, entropy coefficient, value coefficient, and approximate KL calculation. Return `(loss, (pg_loss, v_loss, entropy_loss, approx_kl))`.

Use this current NNX pattern for each minibatch:

```python
optimizer = nnx.Optimizer(agent, tx, wrt=nnx.Param)

@nnx.jit
def train_minibatch(agent, optimizer, obs, actions, old_logprobs,
                    advantages, returns, old_values,
                    clip_coef, ent_coef, vf_coef):
    (loss, metrics), grads = nnx.value_and_grad(
        ppo_loss, argnums=0, has_aux=True
    )(agent, obs, actions, old_logprobs, advantages,
      returns, old_values, clip_coef, ent_coef, vf_coef)
    optimizer.update(agent, grads)
    return loss, metrics
```

This fragment specifies the interface, not a separate script. NNX updates the agent and optimizer in place. Do not assign the return of `optimizer.update` to either object. Do not copy a model into an immutable `TrainState` after the update.

Keep epoch and minibatch iteration in Python. Keep the shuffle key separate from the sampling key. Do not differentiate through observations, stored values, rewards, or old log probabilities.

Remove reassignment such as `actor.apply = jit(actor.apply)`. Remove `.init`, `.apply`, `apply_gradients`, and function fields from the new NNX path. Use `@nnx.jit` at the action and minibatch boundaries.

### 1.5 Define the Phase 1 exit conditions

Before proceeding, make sure that these conditions hold:

1. The module imports without starting a run, when dependencies are available.
2. Agent calls preserve batch dimensions for batch sizes one and four.
3. Both categorical paths return matching log probabilities for the same actions.
4. A synthetic minibatch produces finite loss and changes at least one actor and critic parameter.
5. No Linen or old immutable training-state access remains in `PPO.py`.

Use the installed uv environment for these tests. If a missing dependency blocks execution, fix its declaration and synchronize the environment. Report runtime failures separately from source changes that pass static review.

## Phase 2: Correct Gymnasium collection and return calculations

Estimate: 3–5 hours. Complete this phase in `PPO.py`, then reuse its exact semantics in the other files.

### 2.1 Select same-step reset and stable storage meanings

Construct `SyncVectorEnv` with explicit `AutoresetMode.SAME_STEP`. Reset it once with `seed=args.seed`. Use that seeded observation for both dimension discovery and the first action. Remove the extra unseeded reset.

Change `Storage.dones` to `Storage.episode_ends`. Its new meaning is the end flag for the transition at the same index. Store `terminated | truncated` at `[step]` after calling `envs.step`.

Delete `next_done` from function arguments and return values. Do not retain the old convention that stored the previous transition's end flag before stepping. Update `get_action_and_value`, `rollout`, `compute_gae`, and every caller together.

Keep scalar rollout fields shaped `(num_steps, num_envs)`. Store `episode_ends` as Boolean arrays. Store rewards, values, log probabilities, advantages, and returns as float32 arrays. Observations follow the declared processed space and become float32 before model evaluation.

### 2.2 Correct time-limit returns with one reward correction

Termination means that the environment reaches a terminal state. Truncation means that an external limit ends the episode. Bootstrapping means using the critic's estimate for future rewards. Bootstrap truncations, but never true terminations. [Gymnasium time-limit guide](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)

Use a reward correction instead of adding another next-value buffer. The correction applies only to `truncated & ~terminated`:

```text
training_reward[t, i] = environment_reward[t, i] + gamma * V(final_observation[i])
```

Implement this sequence inside the Python rollout loop:

1. Collect `next_obs, rewards, terminated, truncated, infos` from `envs.step`.
2. Find indices where `truncated & ~terminated` is true.
3. Extract and stack those indices from `infos["final_obs"]`, using the final-observation mask.
4. Evaluate the critic on that batch and add `gamma * final_values` only at those reward indices.
5. Store the corrected rewards and same-transition end flags, then continue from the returned `next_obs`.

Use a normal Python branch when no indices need correction. Do not compile a zero-sized NNX batch. If required final observations or their masks are missing for a truncated transition, raise a clear error. Do not silently bootstrap from the reset observation.

Final observations already passed through each individual environment's observation wrappers. Do not normalize them a second time. Do not replace the returned `next_obs` with final observations because the next action belongs to the new episode.

If termination and truncation are both true, termination wins and no correction applies. Keep raw episode statistics from `RecordEpisodeStatistics`. Do not log corrected rewards as environment returns.

### 2.3 Rebuild GAE around the new buffer semantics

GAE estimates the advantage of an action over the critic's prediction. Evaluate the final rollout observation with the unchanged policy before any optimizer step. Call that array `last_value` and require shape `(num_envs,)`.

Compute advantages with these equations:

```text
next_values = concatenate(values[1:], last_value[None, :], axis=0)
continue_mask = 1 - episode_ends
delta[t] = corrected_rewards[t] + gamma * continue_mask[t] * next_values[t] - values[t]
advantage[t] = delta[t] + gamma * gae_lambda * continue_mask[t] * advantage[t + 1]
returns[t] = advantage[t] + values[t]
```

Initialize the final advantage accumulator to zeros of shape `(num_envs,)`. Use a reverse `jax.lax.scan` over the time axis. Keep this function array-only under `@jax.jit`. Derive the rollout length from input shapes instead of global arguments.

The reward correction supplies the final value for a truncation. The end flag stops the advantage recursion at either kind of episode boundary. Do not also bootstrap a truncated transition through `next_values`. That counts its final value twice.

Use `jax.lax.stop_gradient` on stored targets if the data path requires it. Do not evaluate the critic inside the pure GAE scan. Only the minibatch loss differentiates model parameters.

### 2.4 Read current episode statistics

Use the current batched dictionaries rather than iterating over `infos["final_info"]` as a list. For ended episodes, the relevant structure is:

```python
final_info = infos["final_info"]
episode = final_info["episode"]
mask = infos["_final_info"] & final_info["_episode"]
indices = np.flatnonzero(mask)
```

If `final_info` is absent, skip only the logging block. Always store the transition first. Do not skip rollout collection because no episode ended. If final information lacks episode statistics, leave logging untouched and continue collection.

Use each valid index to read `episode["r"][i]` and `episode["l"][i]`. Convert each value to a Python scalar before logging. Log each completed environment once. Require the expected episode mask when final episode statistics exist.

Keep `RecordEpisodeStatistics` inside each environment factory. That wrapper order preserves unnormalized rewards in the continuous script. Do not add a second vector statistics wrapper. [Gymnasium vector information format](https://gymnasium.farama.org/api/vector/), [synchronous environment reference](https://gymnasium.farama.org/api/vector/sync_vector_env/)

### 2.5 Add exact boundary tests

Use these numerical examples with `gamma=0.9`:

1. True termination: reward `1`, current value `2`, final value `10`. The one-step advantage is `-1` and return is `1`.
2. Truncation only: reward `1`, current value `2`, final value `10`. The corrected reward is `10`, advantage is `8`, and return is `10`.
3. Both flags true: use the true-termination result, with no bootstrap correction.
4. No episode end at the buffer edge: reward `1`, current value `2`, next value `10`. The one-step advantage is `8`.
5. Two episodes in one buffer: a large reward in the second episode must not affect advantages in the first episode.

For the truncation case, use a reset observation with value `100` and a final observation with value `10`. The result must still use `10`. Include a batch with both continuing and ended environments. Test final-info masks with two episode endings on the same step.

## Phase 3: Port the complete pattern to continuous actions and Atari

Estimate: 3–5 hours. Port common changes before adding each file's specialized behavior.

### 3.1 Port continuous NNX models and probabilities

Give the continuous actor explicit `obs_dim` and `action_dim` constructor arguments. Preserve `obs_dim -> 64 -> 64 -> action_dim`, hidden `tanh` activations, and current initializer scales. Define the log standard deviation as `nnx.Param(jnp.zeros((1, action_dim), dtype=jnp.float32))`.

Read this parameter with `self.logstd[...]`. Broadcast it to the mean's shape. Return `(mean, logstd, values)` from the continuous `Agent`. The critic follows Phase 1.

Require a one-dimensional continuous action space in this script. The current continuous distributions assume that layout. Reject unsupported multidimensional action spaces instead of silently changing flattening rules.

Use these Gaussian operations:

```python
std = jnp.exp(logstd)
actions = mean + std * jax.random.normal(
    sample_key, mean.shape, dtype=mean.dtype
)
logprobs = jax.scipy.stats.norm.logpdf(
    actions, loc=mean, scale=std
).sum(axis=-1)
entropy = (logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)).sum(axis=-1)
```

Sum across action dimensions only. Keep sampled actions shaped `(batch, action_dim)` and scalar outputs shaped `(batch,)`. Use the same log-probability expression when evaluating stored actions. [JAX normal log probability](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.stats.norm.logpdf.html)

Preserve the unsquashed normal policy. Preserve `ClipAction` around the environment. Store the original sampled action and its log probability. The wrapper clips only the action delivered to the physical environment. Do not replace stored actions with clipped actions or introduce `tanh` squashing.

### 3.2 Update continuous environment construction

Change the default environment from `Pusher-v4` to `Pusher-v5`. Derive observation and action sizes from the created environment. The Pusher documentation states that v4 is incompatible with MuJoCo 3 and shows v5 as the current interface. Do not promise identical learning curves across environment versions. [Pusher documentation](https://gymnasium.farama.org/environments/mujoco/pusher/)

Preserve this wrapper order, from the base environment outward:

```text
gym.make
optional RecordVideo on environment zero
FlattenObservation
RecordEpisodeStatistics
ClipAction
NormalizeObservation
TransformObservation
NormalizeReward
TransformReward
```

Pass `gamma` explicitly into `make_env`; do not read global arguments. Give `TransformObservation` an explicit float32 `Box` with bounds `-10` and `10` and the flattened observation shape. Its transform clips observations and casts them to float32. The current wrapper requires its `observation_space` argument. [Observation wrapper source](https://gymnasium.farama.org/_modules/gymnasium/wrappers/transform_observation/)

Keep reward normalization and clipping at the current thresholds. Perform the truncation correction after the wrappers return the reward. Keep episode logging tied to the raw statistics inside the wrapper chain.

Implement the currently ignored `capture_video` flag. Request `render_mode="rgb_array"` only for environment zero when recording. Keep recording before observation transforms so video shows rendered frames. Use a directory that includes the run name.

### 3.3 Port the Atari network without changing its architecture

Keep three VALID convolutions followed by a shared 512-unit layer. Define NNX convolution input channels explicitly:

```text
Input: (batch, 84, 84, 1), float32, values in [0, 1]
Conv 1: 1 -> 32 channels, kernel (8, 8), stride (4, 4)
Conv 2: 32 -> 64 channels, kernel (4, 4), stride (2, 2)
Conv 3: 64 -> 64 channels, kernel (3, 3), stride (1, 1)
Flatten: 7 * 7 * 64 = 3136 features
Linear: 3136 -> 512
Actor head: 512 -> action_n
Critic head: 512 -> 1
```

Keep ReLU after each convolution and after the 512-unit layer. Preserve the current orthogonal initializer scales and zero biases. Pass `padding="VALID"` explicitly. NNX's default padding is not a substitute for the current architecture.

Keep the channels-last layout. Do not transpose observations into a PyTorch layout. Do not divide the input by 255 again because `scale_obs=True` already scales it.

Create one Atari `Agent` that owns `Network`, `Actor`, and `Critic`. Compute the shared hidden representation once per agent call. Create one optimizer over the entire agent so the shared network receives both loss terms. Remove the old `network_fn` and `network_params` fields.

Keep one grayscale frame because that is the existing architecture. Do not add frame stacking, life-loss termination, reward-sign clipping, or a FIRE-reset wrapper in this migration.

### 3.4 Update Atari registration and frame skipping

Import `ale_py` inside the Atari startup path before creating environments. Explicit registration can use the current documented `gym.register_envs(ale_py)` call. Importing the Atari Python module for synthetic model tests must not require a running ALE environment. [ALE usage documentation](https://ale.farama.org/)

Change the default ID to `ALE/KungFuMaster-v5`. Construct it with `frameskip=1` and `repeat_action_probability=0.0`. The old `KungFuMasterNoFrameskip-v4` used those values. Current v5 defaults use frame skipping and sticky actions, so accepting the defaults changes the experiment. [KungFuMaster version table](https://ale.farama.org/environments/kung_fu_master/)

Keep `AtariPreprocessing` with `frame_skip=4`, `screen_size=84`, `grayscale_obs=True`, `grayscale_newaxis=True`, and `scale_obs=True`. Keep the existing no-op reset behavior. Do not skip four frames in both the base environment and the preprocessing wrapper.

Use `RecordEpisodeStatistics` before `AtariPreprocessing`, as the current code does. Insert optional video recording near the base environment. Do not switch to ALE's experimental vector environment because this plan uses Gymnasium's explicit same-step reset contract.

### 3.5 Define specialized exit conditions

Make sure that these conditions hold:

1. Both continuous probability paths agree on the same sampled actions, including batches with one item.
2. A continuous synthetic update changes the learned log standard deviation.
3. Atari synthetic inputs produce actor outputs `(batch, action_n)` and values `(batch,)`.
4. An Atari synthetic update changes shared-network parameters and at least one parameter in each head.
5. Factory tests inspect current environment IDs, wrapper arguments, and base frame skipping without creating MuJoCo or ALE instances.

## Phase 4: Finish command handling, logging, and checkpoint saving

Estimate: 2–3 hours. Apply these changes consistently in all scripts.

### 4.1 Fix Boolean flags and invalid batch arguments

Replace `type=bool` for `--track` and `--capture-video` with `argparse.BooleanOptionalAction`. Keep both defaults false. Accept `--track`, `--no-track`, `--capture-video`, and `--no-capture-video`. Reject the old misleading `--track False` syntax.

Validate these argument groups before creating environments:

1. Require positive `num_envs`, `num_steps`, `num_minibatches`, and `update_epochs`.
2. Require `batch_size % num_minibatches == 0` and a nonzero minibatch size.
3. Require `total_timesteps >= batch_size` so at least one update exists.
4. Require finite positive learning rate and gradient norm, plus a finite nonnegative clip coefficient.
5. Require `gamma` and `gae_lambda` in `[0, 1]`.

Use `parser.error` for invalid command-line values. Preserve the existing full-batch interpretation of `num_updates = total_timesteps // batch_size`. Do not add partial final rollouts.

### 4.2 Keep logging independent of optimizer internals

Remove the fixed tuple lookup `agent_state.opt_state[1].hyperparams["learning_rate"]`. Log the schedule value for the rollout that just ran. With the preserved schedule, that count is `(update - 1) * num_minibatches * update_epochs`.

Keep the current metric names. Calculate explained variance with one array library and return `NaN` when the target variance is zero. Avoid Python truth tests on JAX arrays inside compiled functions. Convert scalar metrics to host Python values only at the logging boundary.

Create a unique run name with a timestamp that includes microseconds. Use it for logs, video, and checkpoints. Replace the fixed `videos/{env_id}` and `./model` output targets so separate runs do not collide.

Keep tracking disabled by default. When enabled, preserve `sync_tensorboard=True` and explicit run metadata. Remove `monitor_gym=True` because this project uses Gymnasium recording rather than OpenAI Gym patching. Do not claim that `monitor_gym` itself is removed. Do not add network-based tests. [W&B initialization source](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_init.py)

### 4.3 Save NNX state instead of Python model objects

Delete the `orbax_utils` import and the old checkpointer construction. Save only array state, primitive metadata, and explicit random-key data. Do not save callbacks, writers, environments, or NNX graph definitions.

Use a payload with these entries:

```python
payload = {
    "model": nnx.state(agent),
    "optimizer": nnx.state(optimizer),
    "global_step": np.asarray(global_step, dtype=np.int64),
    "action_key": jax.random.key_data(action_key),
    "permutation_key": jax.random.key_data(permutation_key),
}
```

Use JSON-compatible `custom_metadata` for `vars(args)`, a schema label such as `"nnx-v1"`, and random-key implementation names. Restore saved keys with `jax.random.wrap_key_data` and the recorded implementation when testing restoration.

Resolve the checkpoint directory to an absolute path. Use a fresh run-specific final directory under `models/`. Create its parent, but let Orbax create the final checkpoint directory. Keep overwrite disabled. Do not delete old checkpoints or add `force=True`.

NNX state is a pytree, a nested structure of arrays. For restoration tests, construct the same agent and optimizer structure. Supply an abstract target tree with matching shapes, dtypes, and NNX variable types to `ocp.load`. Apply restored state with `nnx.update`. Compare parameters, optimizer steps, and predictions. [Flax checkpoint guide](https://flax.readthedocs.io/en/latest/guides/checkpointing.html)

Use the array-only payload above to create the restoration target:

```python
target = jax.tree.map(
    lambda x: np.zeros_like(x) if isinstance(x, np.ndarray)
    else jax.ShapeDtypeStruct(x.shape, x.dtype), payload
)
restored = ocp.load(checkpoint_dir, abstract_state=target)
nnx.update(agent, restored["model"])
nnx.update(optimizer, restored["optimizer"])
```

Keep NumPy leaves as NumPy restoration targets. Orbax's JAX abstract targets request sharding information that NumPy scalar checkpoints do not contain. In the test, mutate the live model and optimizer after saving. Make sure that loading restores the saved values. Build `target` before those mutations. Reconstruct each sampling key from its saved key data and recorded implementation.

This pass does not add a resume command or a converter for old Linen checkpoints. It also does not save live environment or normalization-wrapper state. Add this one-line limit near checkpoint creation:

```python
# shortcut: environment and wrapper state are not saved; include them before adding resume or standalone evaluation.
```

Do not describe this payload as an exact resumable experiment. Restoring model predictions on the same synthetic inputs is the acceptance test. Real continuous-policy evaluation also requires the original observation normalization state.

### 4.4 Keep cleanup and failures predictable

Use `try/finally` to close all created resources. If enabled, finish the W&B run. Keep signal registration inside `main`. Do not save from inside a compiled function.

Save after a normal run or a handled `KeyboardInterrupt`, once the agent and optimizer exist. Do not save after an unexpected training exception. Keep the original exception visible rather than replacing it with an unrelated checkpoint failure.

Put checkpoint saving inside the resource cleanup boundary so a save error still closes environments and writers. Let a save error surface. Do not catch it and print a successful completion message.

Keep startup resource creation inside the same cleanup structure. If writer or tracking initialization fails after environments exist, close the environments. Avoid cleanup paths that reference an uninitialized writer or tracking run.

### 4.5 Audit removed and deprecated access

Search the three implementations for these patterns:

```text
flax.linen, nn.compact, FrozenDict, TrainState, AgentParams, AgentState
tensorflow_probability, tfp., PRNGKeyArray, random.PRNGKey(
orbax_utils, PyTreeCheckpointer, save_args_from_target
apply_gradients, actor.apply, critic.apply, network.apply
type=bool, monitor_gym, .value, .raw_value
```

Review each occurrence rather than deleting unrelated text automatically. `storage.values` is valid and is not deprecated variable access. `get_action_and_value` is a valid function name and is not `.value` access.

Do not suppress warnings to claim a clean migration. When runtime checks become possible, treat deprecation warnings from these code paths as failures. Report warnings from external packages separately with their source location.

## Phase 5: Add acceptance checks and a precise handoff

Estimate: 1–2 hours. This phase does not include training.

### 5.1 Add a dependency-free static test group

Use Python's built-in `ast` and `unittest` modules. Make this group runnable without JAX, Flax, Gymnasium, or Optax. Parse source rather than importing the training scripts.

Make sure that these conditions hold:

1. Each training file parses and exposes `main` and `parse_args`.
2. Removed imports and interfaces from Phase 4 are absent from executable source.
3. No environment, writer, tracking run, or checkpoint save executes at module import time.
4. Each environment construction selects explicit same-step reset.
5. `pyproject.toml` declares the intended Python target, core dependencies, and optional feature groups.

Use AST nodes for imports and deprecated attribute access where practical. Avoid fragile assertions about formatting or exact line numbers. Manual review remains necessary for data flow and numerical correctness.

### 5.2 Add synthetic numerical test groups

Use one test file, `tests/test_modernization.py`, with small fixtures. Use standard-library mocking for writers and factories. Load each script once per test group. Do not add pytest or test helper packages.

Cover these groups:

1. Models and optimizer: batch sizes one and four, finite outputs, correct shapes, a changed parameter in each required subnetwork, and optimizer step increments.
2. Categorical distribution: zero logits give log probability `-log(action_n)` and entropy `log(action_n)`. Use extreme finite logits to test numerical stability.
3. Gaussian distribution: zero mean and unit variance give log probability `-action_dim * log(2*pi)/2` at zero. Entropy is `action_dim * log(2*pi*e)/2`.
4. Randomness: identical input keys reproduce samples, returned keys advance, and action sampling does not reuse the shuffle stream.
5. Targets and loss: use all boundary cases from Phase 2, finite scalar loss, correct gradient coverage, and no gradient path through collected targets.

Construct synthetic data directly. Do not run the scripts' training loops. Test the continuous log-standard-deviation gradient with data that produces a nonzero gradient, rather than assuming any random batch does so.

For Atari, use synthetic `(batch, 84, 84, 1)` observations. An expensive orthogonal initialization is not a training experiment, but keep the batch small. Do not instantiate ALE or require ROMs for a network test.

### 5.3 Add synthetic environment and save tests

Define a tiny local `gym.Env` that emits distinct final and reset observations. Give it one-step and multi-step episode variants. It must support deterministic termination and deterministic truncation.

Cover these groups:

1. Collection: same-step reset, mixed end flags, correct final observation, and one log entry for each finished environment.
2. Continuous wrappers: float32 observations stay inside the declared clipped space, raw episode returns remain raw, and stored actions remain unclipped.
3. Factory arguments: current MuJoCo and Atari IDs, ALE base `frameskip=1`, preprocessing `frame_skip=4`, and recording only on environment zero.
4. Checkpoints: save and load a tiny NNX agent and optimizer in a temporary directory. Compare state and reject a second save to that final directory.
5. Cleanup: normal completion, interruption, unexpected exception, and save failure close created resources without replacing the original training error.

Mock video and tracking components in the synthetic tests. Do not call W&B or create actual videos. Synthetic environment tests can perform a few fixed steps, but they must not optimize a policy or claim learning performance.

After the full environment installs, add a separate bounded environment smoke test. A smoke test establishes that a component starts and performs basic operations. Create the default Acrobot, Pusher, and ALE environments through the actual factories, with recording and tracking disabled.

Seed each environment, reset it, perform four valid actions, inspect output shapes and finite rewards, and close it. Use no policy updates or training loops. Keep synthetic tests responsible for exact episode-boundary cases. Report an actual MuJoCo or ALE failure rather than skipping it when the corresponding extra is installed.

### 5.4 Run checks through the locked uv environment

For a dependency-free syntax pass from the repository root, use:

```bash
python3 -B - <<'PY'
import ast
from pathlib import Path

paths = [Path(name) for name in (
    "PPO.py", "PPO_continuous_action.py", "PPO_atari.py",
    "tests/test_modernization.py",
)]
for path in paths:
    ast.parse(path.read_text(), filename=str(path))
    print(f"Parsed {path}")
PY
```

The command reads source without importing dependencies or creating bytecode. It remains available if environment setup fails. Run the static unittest group next. Keep that class independent of optional numerical imports.

After Phase 0 succeeds, run all synthetic tests and bounded environment smoke tests through uv. Missing core dependencies are setup failures. Optional tests can skip only in a deliberately core-only environment. With `--all-extras`, report unavailable feature dependencies as failures rather than silently skipping their tests.

Run tests with deprecation warnings promoted to errors. Use this command only after the tests exist:

```bash
uv run --locked --all-extras python -B \
    -W error::DeprecationWarning -W error::FutureWarning \
    -m unittest discover -s tests -p 'test_modernization.py' -v
```

Run `uv lock --check` after the final dependency edits. Make sure that the README's core and feature sync commands match the manifest. Make sure that `.python-version` and `uv.lock` are not ignored. Use uv to fix missing declared dependencies, but do not install undeclared packages into `.venv` with pip.

Do not run training, contact W&B, or claim GPU support from these tests. If a dependency failure remains, record its exact error and the tests that it blocks. Preserve static-test results separately from runtime results.

### 5.5 Report completion without claiming training success

The final implementation report must state these facts:

1. Which of the three scripts now use native NNX, current Gymnasium behavior, and current saving interfaces.
2. Which static checks passed and which synthetic tests actually executed.
3. Which dependency installs or tests failed or skipped, with the exact reason for each group.
4. The Python patch release, resolved direct dependencies, and successful locked installation commands. State that no training ran.
5. Which real environment smoke tests passed. State that learning performance, actual video output, online tracking, and GPU execution remain untested.

Do not close the original compatibility issue solely because source changes parse. Establish the locked dependency set and execute the acceptance tests before claiming installation and execution compatibility. Learning experiments remain separate from these checks.

## Start instruction for the implementing model

Implement Phase 0 first. Create the uv project, install standard Python 3.14, declare and install dependencies, and record the actual library versions. Read `PPO.py` before changing its imports. Do not patch the old scripts just to make their legacy imports pass during setup.

Continue through Phases 1–5 in order. Keep the single-file design and preserve PPO coefficients. Add the relevant tests as the numerical functions change. Run all available acceptance tests in the locked uv environment.

Do not leave the continuous and Atari scripts on the old interfaces when declaring the overall migration complete. Dependency setup is in scope. Training experiments, online tracking, GPU work, and unrelated refactoring remain outside this plan.
