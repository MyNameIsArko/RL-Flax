import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tensorboardX import SummaryWriter
from tqdm import tqdm


class QNetwork(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, *, rngs: nnx.Rngs) -> None:
        self.layers = nnx.Sequential(
            nnx.Linear(obs_dim, 120, rngs=rngs), nnx.relu,
            nnx.Linear(120, 84, rngs=rngs), nnx.relu,
            nnx.Linear(84, action_dim, rngs=rngs),
        )

    def __call__(self, observations: np.ndarray | jax.Array) -> jax.Array:
        return self.layers(observations)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', default='CartPole-v1', help='Gymnasium environment identifier')
    parser.add_argument('--total-timesteps', type=int, default=500000, help='number of environment steps')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4, help='Adam learning rate')
    parser.add_argument('--buffer-size', type=int, default=10000, help='maximum replay transitions')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount for future rewards')
    parser.add_argument('--batch-size', type=int, default=128, help='replay samples per update')
    parser.add_argument('--learning-starts', type=int, default=10000, help='minimum steps before learning')
    parser.add_argument('--train-frequency', type=int, default=10, help='environment steps between updates')
    parser.add_argument('--target-network-frequency', type=int, default=500,
                        help='environment steps between target copies')
    parser.add_argument('--start-e', type=float, default=1.0, help='initial random-action probability')
    parser.add_argument('--end-e', type=float, default=0.05, help='final random-action probability')
    parser.add_argument('--exploration-fraction', type=float, default=0.5,
                        help='fraction of training used to reduce exploration')
    parser.add_argument('--seed', type=int, default=1, help='seed for model, environment, and replay')
    parser.add_argument('--exp-name', default='DQN', help='experiment name prefix')
    args = parser.parse_args(argv)

    for name in ('total-timesteps', 'buffer-size', 'batch-size', 'train-frequency',
                 'target-network-frequency'):
        if getattr(args, name.replace('-', '_')) <= 0:
            parser.error(f'--{name} must be positive')
    if args.batch_size > args.buffer_size:
        parser.error('--batch-size must not exceed --buffer-size')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('--learning-rate must be finite and positive')
    if not math.isfinite(args.gamma) or not 0 <= args.gamma <= 1:
        parser.error('--gamma must be finite and in [0, 1]')
    if (not all(math.isfinite(value) for value in (args.start_e, args.end_e))
            or not 0 <= args.end_e <= args.start_e <= 1):
        parser.error('exploration probabilities must satisfy 0 <= --end-e <= --start-e <= 1')
    if (not math.isfinite(args.exploration_fraction)
            or not 0 < args.exploration_fraction <= 1):
        parser.error('--exploration-fraction must be finite and in (0, 1]')
    if args.learning_starts < 0:
        parser.error('--learning-starts must be nonnegative')
    earliest_update = max(args.learning_starts, args.batch_size)
    first_update_step = ((earliest_update + args.train_frequency - 1) // args.train_frequency) * args.train_frequency
    if first_update_step > args.total_timesteps:
        parser.error('settings must allow at least one gradient update')
    return args


@nnx.jit
def greedy_action(q_network: QNetwork, observation: jax.Array) -> jax.Array:
    return jnp.argmax(q_network(observation[None, :])[0])


@nnx.jit
def train_step(
    q_network: QNetwork,
    target_network: QNetwork,
    optimizer: nnx.Optimizer,
    observations: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_observations: jax.Array,
    terminations: jax.Array,
    gamma: float,
) -> tuple[jax.Array, jax.Array]:
    next_q_values = jnp.max(target_network(next_observations), axis=-1)
    targets = rewards + gamma * (1 - terminations.astype(jnp.float32)) * next_q_values
    targets = jax.lax.stop_gradient(targets)

    def loss_fn(network: QNetwork) -> tuple[jax.Array, jax.Array]:
        q_values = network(observations)
        selected_q_values = jnp.take_along_axis(q_values, actions[:, None], axis=-1)[:, 0]
        return jnp.mean((selected_q_values - targets) ** 2), jnp.mean(selected_q_values)

    (loss, mean_q), grads = nnx.value_and_grad(loss_fn, has_aux=True)(q_network)
    optimizer.update(q_network, grads)
    return loss, mean_q


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_name = f'{args.exp_name}__{args.seed}__{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
    env: gym.Env | None = None
    writer: SummaryWriter | None = None
    try:
        env = gym.wrappers.RecordEpisodeStatistics(gym.make(args.env_id))
        if not isinstance(env.observation_space, gym.spaces.Box) or len(env.observation_space.shape) != 1:
            raise ValueError('DQN requires a one-dimensional Box observation space')
        if not isinstance(env.action_space, gym.spaces.Discrete) or env.action_space.start != 0:
            raise ValueError('DQN requires a zero-based Discrete action space')

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        obs, _ = env.reset(seed=args.seed)
        rng = np.random.default_rng(args.seed)
        q_network = QNetwork(obs_dim, action_dim, rngs=nnx.Rngs(args.seed))
        target_network = nnx.clone(q_network)
        optimizer = nnx.Optimizer(q_network, optax.adam(args.learning_rate), wrt=nnx.Param)
        writer = SummaryWriter(f'runs/{args.env_id}/{run_name}')
        observations = np.empty((args.buffer_size, obs_dim), dtype=np.float32)
        next_observations = np.empty_like(observations)
        actions = np.empty(args.buffer_size, dtype=np.int32)
        rewards = np.empty(args.buffer_size, dtype=np.float32)
        terminations = np.empty(args.buffer_size, dtype=np.bool_)
        exploration_steps = max(1, int(args.exploration_fraction * args.total_timesteps))
        start_time = time.time()

        for global_step in tqdm(range(1, args.total_timesteps + 1)):
            progress = min((global_step - 1) / exploration_steps, 1.0)
            epsilon = args.start_e + (args.end_e - args.start_e) * progress
            action = (int(rng.integers(action_dim)) if rng.random() < epsilon
                      else int(greedy_action(q_network, jnp.asarray(obs, jnp.float32))))
            next_obs, reward, terminated, truncated, info = env.step(action)
            buffer_index = (global_step - 1) % args.buffer_size
            observations[buffer_index] = obs
            next_observations[buffer_index] = next_obs
            actions[buffer_index] = action
            rewards[buffer_index] = reward
            terminations[buffer_index] = terminated

            if terminated or truncated:
                episode = info['episode']
                writer.add_scalar('charts/episodic_return', float(episode['r']), global_step)
                writer.add_scalar('charts/episodic_length', int(episode['l']), global_step)
                obs, _ = env.reset()
            else:
                obs = next_obs

            replay_size = min(global_step, args.buffer_size)
            if (global_step >= args.learning_starts and replay_size >= args.batch_size
                    and global_step % args.train_frequency == 0):
                indices = rng.integers(0, replay_size, size=args.batch_size)
                loss, mean_q = train_step(
                    q_network, target_network, optimizer,
                    jnp.asarray(observations[indices]), jnp.asarray(actions[indices]),
                    jnp.asarray(rewards[indices]), jnp.asarray(next_observations[indices]),
                    jnp.asarray(terminations[indices]), args.gamma,
                )
                writer.add_scalar('losses/td_loss', float(loss), global_step)
                writer.add_scalar('losses/q_values', float(mean_q), global_step)
            if global_step % args.target_network_frequency == 0:
                nnx.update(target_network, nnx.state(q_network, nnx.Param))
            if global_step % 100 == 0 or global_step == args.total_timesteps:
                writer.add_scalar('charts/epsilon', epsilon, global_step)
                writer.add_scalar('charts/SPS', int(global_step / (time.time() - start_time)), global_step)

        # shortcut: save policy weights only; include replay and optimizer state before adding training resume.
        from orbax.checkpoint import v1 as ocp
        checkpoint_path = (Path('models') / run_name).resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        ocp.save(checkpoint_path, nnx.state(q_network), custom_metadata={
            'config': vars(args), 'global_step': args.total_timesteps,
        })
        print('Training complete!')
    finally:
        if env is not None:
            env.close()
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    main()
