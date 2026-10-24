import ast
import importlib
import math
import tomllib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ('PPO.py', 'PPO_continuous_action.py', 'PPO_atari.py')


class StaticAcceptanceTests(unittest.TestCase):
    def test_scripts_parse_and_export_entrypoints(self):
        for name in SCRIPTS:
            tree = ast.parse((ROOT / name).read_text(), filename=name)
            definitions = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            self.assertTrue({'main', 'parse_args'} <= definitions, name)

    def test_removed_apis_are_absent(self):
        removed = {'FrozenDict', 'TrainState', 'AgentParams', 'AgentState', 'PRNGKeyArray',
                   'PyTreeCheckpointer', 'save_args_from_target'}
        for name in SCRIPTS:
            tree = ast.parse((ROOT / name).read_text(), filename=name)
            source = ast.unparse(tree)
            for item in removed | {'tensorflow_probability', 'orbax_utils'}:
                self.assertNotIn(item, source, (name, item))
            attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            self.assertNotIn('value', attributes, name)
            self.assertNotIn('raw_value', attributes, name)
            self.assertNotIn('type=bool', source, name)
            self.assertNotIn('monitor_gym', source, name)
            self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'SyncVectorEnv' and any(k.arg == 'autoreset_mode' for k in n.keywords)
                for n in ast.walk(tree)), name)

    def test_import_has_no_top_level_calls(self):
        for name in SCRIPTS:
            tree = ast.parse((ROOT / name).read_text(), filename=name)
            for node in tree.body:
                if isinstance(node, ast.Expr):
                    self.assertFalse(isinstance(node.value, ast.Call), (name, ast.unparse(node)))

    def test_project_configuration(self):
        project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        self.assertEqual(project['project']['requires-python'], '>=3.14')
        self.assertEqual(set(project['project']['optional-dependencies']),
                         {'mujoco', 'atari', 'video', 'tracking'})
        self.assertTrue(any(d.startswith('jax') for d in project['project']['dependencies']))
        self.assertEqual((ROOT / '.python-version').read_text().strip(), '3.14')
        self.assertTrue((ROOT / 'uv.lock').exists())

    def test_environment_factory_configuration(self):
        defaults = {}
        for name in SCRIPTS:
            tree = ast.parse((ROOT / name).read_text())
            parser_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_args')
            env_arg = next(n for n in ast.walk(parser_fn) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
                and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == '--env-id')
            defaults[name] = ast.literal_eval(next(k.value for k in env_arg.keywords if k.arg == 'default'))
        self.assertEqual(defaults, {'PPO.py': 'Acrobot-v1',
            'PPO_continuous_action.py': 'Pusher-v5', 'PPO_atari.py': 'ALE/KungFuMaster-v5'})
        atari_tree = ast.parse((ROOT / 'PPO_atari.py').read_text())
        atari_factory = next(n for n in atari_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_env')
        calls = [n for n in ast.walk(atari_factory) if isinstance(n, ast.Call)]
        make = next(n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'make')
        self.assertEqual({k.arg: ast.literal_eval(k.value) for k in make.keywords if k.arg in
                          {'frameskip', 'repeat_action_probability'}},
                         {'frameskip': 1, 'repeat_action_probability': 0.0})
        preprocessing = next(n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'AtariPreprocessing')
        self.assertEqual({k.arg: ast.literal_eval(k.value) for k in preprocessing.keywords},
            {'frame_skip': 4, 'screen_size': 84, 'grayscale_obs': True,
             'grayscale_newaxis': True, 'scale_obs': True})
        for name in SCRIPTS:
            self.assertIn('RecordVideo', (ROOT / name).read_text())


class NumericalAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jax = importlib.import_module('jax')
        cls.jnp = importlib.import_module('jax.numpy')
        cls.optax = importlib.import_module('optax')
        cls.nnx = importlib.import_module('flax.nnx')
        cls.discrete = importlib.import_module('PPO')
        cls.continuous = importlib.import_module('PPO_continuous_action')
        cls.atari = importlib.import_module('PPO_atari')

    def test_boolean_flags_and_argument_validation(self):
        for module in (self.discrete, self.continuous, self.atari):
            self.assertFalse(module.parse_args([]).track)
            self.assertTrue(module.parse_args(['--track']).track)
            self.assertFalse(module.parse_args(['--no-track']).track)
            with self.assertRaises(SystemExit):
                module.parse_args(['--track', 'False'])
            with self.assertRaises(SystemExit):
                module.parse_args(['--num-envs', '0'])
            with self.assertRaises(SystemExit):
                module.parse_args(['--learning-rate', 'nan'])

    def test_model_batch_shapes_and_ppo_updates(self):
        jax, jnp, nnx = self.jax, self.jnp, self.nnx
        for batch in (1, 4):
            agent = self.discrete.Agent(3, 2, rngs=nnx.Rngs(default=jax.random.key(batch)))
            logits, values = agent(jnp.ones((batch, 3)))
            self.assertEqual(logits.shape, (batch, 2))
            self.assertEqual(values.shape, (batch,))
            self.assertTrue(bool(jnp.all(jnp.isfinite(logits))))
            self.assertTrue(bool(jnp.all(jnp.isfinite(values))))
        agent = self.discrete.Agent(3, 2, rngs=nnx.Rngs(default=jax.random.key(20)))
        optimizer = nnx.Optimizer(agent, self.optax.adamw(1e-3), wrt=nnx.Param)
        obs = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10
        actions = jnp.array([0, 1, 0, 1], jnp.int32)
        logprobs = self.discrete.get_action_and_value2(agent, obs, actions)[0]
        before_actor = [x.copy() for x in jax.tree.leaves(nnx.state(agent.actor))]
        before_critic = [x.copy() for x in jax.tree.leaves(nnx.state(agent.critic))]
        loss, _ = self.discrete.train_minibatch(agent, optimizer, obs, actions, logprobs,
            jnp.array([-1., 0., 1., 2.]), jnp.array([1., 2., 3., 4.]), jnp.zeros(4),
            0.2, 0.01, 0.5)
        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertEqual(int(optimizer.step[...]), 1)
        self.assertTrue(any(bool(jnp.any(a != b)) for a, b in zip(before_actor, jax.tree.leaves(nnx.state(agent.actor)))))
        self.assertTrue(any(bool(jnp.any(a != b)) for a, b in zip(before_critic, jax.tree.leaves(nnx.state(agent.critic)))))

    def test_categorical_values_and_extreme_logit_gradients(self):
        jax, jnp, nnx = self.jax, self.jnp, self.nnx
        agent = self.discrete.Agent(1, 3, rngs=self.nnx.Rngs(default=jax.random.key(5)))
        class FixedLogits(nnx.Module):
            def __init__(self, logits): self.logits = nnx.Param(logits)
            def __call__(self, obs):
                return jnp.broadcast_to(self.logits[...], (obs.shape[0], 3)), jnp.zeros((obs.shape[0],))
        fixed = FixedLogits(jnp.zeros(3))
        obs = jnp.ones((1, 1))
        zero_lp, zero_ent, _ = self.discrete.get_action_and_value2(fixed, obs, jnp.array([0]))
        self.assertTrue(bool(jnp.allclose(zero_lp, jnp.array([-math.log(3)]))))
        self.assertTrue(bool(jnp.allclose(zero_ent, math.log(3))))
        fixed.logits[...] = jnp.array([1000., -1000., 0.])
        grad = nnx.grad(lambda model: self.discrete.get_action_and_value2(
            model, obs, jnp.array([0]))[1].sum())(fixed).logits[...]
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
        self.assertTrue(bool(jnp.allclose(grad, 0)))
        atari_fixed = FixedLogits(jnp.array([1000., -1000., 0.]))
        atari_grad = nnx.grad(lambda model: self.atari.get_action_and_value2(
            model, obs, jnp.array([0]))[1].sum())(atari_fixed).logits[...]
        self.assertTrue(bool(jnp.all(jnp.isfinite(atari_grad))))
        sampled, lp, values, key2 = self.discrete.get_action_and_value(agent, jnp.ones((1, 1)), jax.random.key(9))
        checked, ent, _ = self.discrete.get_action_and_value2(agent, jnp.ones((1, 1)), sampled)
        self.assertEqual(sampled.dtype, jnp.int32)
        self.assertTrue(bool(jnp.allclose(lp, checked)))
        self.assertEqual((lp.shape, ent.shape, values.shape), ((1,), (1,), (1,)))
        self.assertFalse(bool(jnp.array_equal(jax.random.key_data(key2), jax.random.key_data(jax.random.key(9)))))

    def test_continuous_action_space_must_be_one_dimensional(self):
        import gymnasium as gym
        import numpy as np
        envs = mock.Mock()
        envs.single_action_space = gym.spaces.Box(-1, 1, shape=(2, 2), dtype=np.float32)
        envs.close = mock.Mock()
        with mock.patch.object(self.continuous.gym.vector, 'SyncVectorEnv', return_value=envs), \
             mock.patch('sys.argv', ['PPO_continuous_action.py', '--num-steps', '1',
                 '--num-minibatches', '1', '--total-timesteps', '1']):
            with self.assertRaisesRegex(ValueError, 'one-dimensional Box action space'):
                self.continuous.main()
        envs.close.assert_called_once()

    def test_gaussian_formulas_sampling_and_logstd_update(self):
        jax, jnp, nnx = self.jax, self.jnp, self.nnx
        zero_mean, zero_logstd, zero_action = jnp.zeros((1, 2)), jnp.zeros((1, 2)), jnp.zeros((1, 2))
        lp = self.continuous.normal_logprob(zero_mean, zero_logstd, zero_action)
        self.assertTrue(bool(jnp.allclose(lp, -math.log(2 * math.pi))))
        agent = self.continuous.Agent(3, 2, rngs=nnx.Rngs(default=jax.random.key(30)))
        optimizer = nnx.Optimizer(agent, self.optax.adamw(1e-2), wrt=nnx.Param)
        obs = jnp.array([[0.2, -0.3, 0.5], [0.8, 0.1, -0.4], [-0.5, 0.2, 0.7], [0.1, 0.4, 0.9]])
        acts, logp, vals, _ = self.continuous.get_action_and_value(agent, obs, jax.random.key(31))
        logp2, entropy, _ = self.continuous.get_action_and_value2(agent, obs, acts)
        self.assertEqual((acts.shape, logp.shape, vals.shape, entropy.shape), ((4, 2), (4,), (4,), (4,)))
        self.assertTrue(bool(jnp.allclose(logp, logp2)))
        self.assertTrue(bool(jnp.allclose(entropy, math.log(2 * math.pi * math.e))))
        before = agent.actor.logstd[...].copy()
        self.continuous.train_minibatch(agent, optimizer, obs, acts, logp,
            jnp.array([-2., -1., 1., 3.]), jnp.array([1., -1., 2., 0.]), vals,
            0.2, 0.0, 0.5)
        self.assertTrue(bool(jnp.any(before != agent.actor.logstd[...])) )

    def test_atari_shapes_and_shared_network_update(self):
        jax, jnp, nnx = self.jax, self.jnp, self.nnx
        agent = self.atari.Agent(3, rngs=nnx.Rngs(default=jax.random.key(40)))
        obs = jnp.ones((1, 84, 84, 1), jnp.float32)
        logits, values = agent(obs)
        self.assertEqual((logits.shape, values.shape), ((1, 3), (1,)))
        optimizer = nnx.Optimizer(agent, self.optax.adamw(1e-4), wrt=nnx.Param)
        batch = jnp.concatenate((obs, obs * 0.5), axis=0)
        acts = jnp.array([0, 2], jnp.int32)
        lp = self.atari.get_action_and_value2(agent, batch, acts)[0]
        before = [jax.tree.leaves(nnx.state(m)) for m in (agent.network, agent.actor, agent.critic)]
        self.atari.train_minibatch(agent, optimizer, batch, acts, lp, jnp.array([-1., 1.]),
            jnp.array([1., 2.]), jnp.zeros(2), 0.1, 0.01, 0.5)
        for old, module in zip(before, (agent.network, agent.actor, agent.critic)):
            new = jax.tree.leaves(nnx.state(module))
            self.assertTrue(any(bool(jnp.any(a != b)) for a, b in zip(old, new)))

    def test_gae_episode_boundaries_and_final_value(self):
        jnp = self.jnp
        # True termination ignores the following value; a truncation is encoded by its corrected reward.
        adv, ret = self.discrete.compute_gae(jnp.array([[1.]]), jnp.array([[2.]]),
            jnp.array([[True]]), jnp.array([10.]), 0.9, 0.95)
        self.assertTrue(bool(jnp.allclose(adv, jnp.array([[-1.]]))))
        self.assertTrue(bool(jnp.allclose(ret, jnp.array([[1.]]))))
        adv, ret = self.discrete.compute_gae(jnp.array([[10.]]), jnp.array([[2.]]),
            jnp.array([[True]]), jnp.array([100.]), 0.9, 0.95)
        self.assertTrue(bool(jnp.allclose(adv, jnp.array([[8.]]))))
        self.assertTrue(bool(jnp.allclose(ret, jnp.array([[10.]]))))
        adv, _ = self.discrete.compute_gae(jnp.array([[1.]]), jnp.array([[2.]]),
            jnp.array([[False]]), jnp.array([10.]), 0.9, 0.95)
        self.assertTrue(bool(jnp.allclose(adv, jnp.array([[8.]]))))
        adv, _ = self.discrete.compute_gae(jnp.array([[1.], [100.]]), jnp.array([[2.], [2.]]),
            jnp.array([[True], [False]]), jnp.array([0.]), 0.9, 0.95)
        self.assertTrue(bool(jnp.allclose(adv[0], jnp.array([-1.]))))
        target_grad = self.jax.grad(lambda rewards: self.discrete.compute_gae(
            rewards, jnp.array([[2.]]), jnp.array([[True]]), jnp.array([0.]), 0.9, 0.95)[1].sum())(
                jnp.array([[1.]]))
        self.assertTrue(bool(jnp.allclose(target_grad, 0)))

    def test_rng_repeatability_and_independent_keys(self):
        jax, jnp, nnx = self.jax, self.jnp, self.nnx
        agent = self.discrete.Agent(2, 3, rngs=nnx.Rngs(default=jax.random.key(51)))
        obs = jnp.ones((4, 2))
        k1 = jax.random.key(52)
        a1 = self.discrete.get_action_and_value(agent, obs, k1)
        a2 = self.discrete.get_action_and_value(agent, obs, k1)
        self.assertTrue(bool(jnp.array_equal(a1[0], a2[0])))
        self.assertTrue(bool(jnp.array_equal(a1[3], a2[3])))
        action_key, shuffle_key = jax.random.split(jax.random.key(53))
        action_key2 = self.discrete.get_action_and_value(agent, obs, action_key)[3]
        shuffle = jax.random.permutation(shuffle_key, 4)
        self.assertFalse(bool(jnp.array_equal(jax.random.key_data(action_key2), jax.random.key_data(shuffle_key))))
        self.assertEqual(shuffle.shape, (4,))


class CheckpointTests(unittest.TestCase):
    def test_orbax_v1_restores_model_optimizer_and_rejects_overwrite(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        import optax
        from flax import nnx
        from orbax.checkpoint import v1 as ocp
        import PPO
        agent = PPO.Agent(3, 2, rngs=nnx.Rngs(default=jax.random.key(60)))
        optimizer = nnx.Optimizer(agent, optax.adamw(1e-3), wrt=nnx.Param)
        obs = jnp.ones((2, 3))
        saved_prediction = agent(obs)
        # Advance optimizer state before the save so restoration covers both trees.
        obs = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        actions = jnp.array([0, 1], jnp.int32)
        old_lp = PPO.get_action_and_value2(agent, obs, actions)[0]
        PPO.train_minibatch(agent, optimizer, obs, actions, old_lp, jnp.array([-1., 1.]),
            jnp.array([1., 2.]), jnp.zeros(2), 0.2, 0.01, 0.5)
        saved_prediction = agent(obs)
        saved_step = int(optimizer.step[...])
        action_key, permutation_key = jax.random.key(61), jax.random.key(62)
        payload = {'model': nnx.state(agent), 'optimizer': nnx.state(optimizer),
            'global_step': np.asarray(7, np.int64), 'action_key': jax.random.key_data(action_key),
            'permutation_key': jax.random.key_data(permutation_key)}
        target = jax.tree.map(lambda x: np.zeros_like(x) if isinstance(x, np.ndarray)
                              else jax.ShapeDtypeStruct(x.shape, x.dtype), payload)
        # Orbax 0.12.8 omits scalar sharding metadata for NumPy scalar leaves.
        target['global_step'] = np.asarray(0, np.int64)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint'
            ocp.save(path, payload, custom_metadata={'schema': 'nnx-v1'})
            with self.assertRaisesRegex(ValueError, 'already exists'):
                ocp.save(path, payload, custom_metadata={'schema': 'nnx-v1'})
            # Change a model value after saving; loading must restore the saved prediction.
            agent.actor.layers.layers[0].bias[...] = 99
            optimizer.step[...] = 99
            restored = ocp.load(path, abstract_state=target)
            nnx.update(agent, restored['model'])
            nnx.update(optimizer, restored['optimizer'])
            actual = agent(obs)
            self.assertTrue(bool(jnp.allclose(saved_prediction[0], actual[0])))
            self.assertEqual(int(restored['global_step']), 7)
            self.assertEqual(int(optimizer.step[...]), saved_step)
            self.assertTrue(bool(jnp.array_equal(restored['action_key'], jax.random.key_data(action_key))))
            self.assertTrue(bool(jnp.array_equal(restored['permutation_key'], jax.random.key_data(permutation_key))))
            self.assertEqual(jax.random.key_impl(jax.random.wrap_key_data(
                restored['action_key'], impl='threefry2x32')), jax.random.key_impl(jax.random.key(61)))


class SyntheticEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gymnasium as gym
        cls.gym = gym

    class BoundaryEnv:
        def __new__(cls, end):
            import gymnasium as gym
            import numpy as np
            class Env(gym.Env):
                observation_space = gym.spaces.Box(-200, 200, shape=(1,), dtype=np.float32)
                action_space = gym.spaces.Discrete(2)
                def __init__(self): self.end = end
                def reset(self, *, seed=None, options=None):
                    super().reset(seed=seed)
                    self.steps = 0
                    return np.array([1.0], np.float32), {}
                def step(self, action):
                    self.steps += 1
                    truncated = self.end in ('truncated', 'both') and self.steps == 1
                    terminated = self.end in ('terminated', 'both') and self.steps == 1
                    final = 10.0 if truncated else 20.0 if terminated else 2.0
                    return np.array([final], np.float32), 1.0, terminated, truncated, {}
            return Env()

    def test_rollout_bootstraps_final_observation_and_logs_masked_episodes(self):
        import jax
        import jax.numpy as jnp
        import PPO
        from flax import nnx
        envs = self.gym.vector.SyncVectorEnv([
            lambda: self.gym.wrappers.RecordEpisodeStatistics(self.BoundaryEnv('truncated')),
            lambda: self.gym.wrappers.RecordEpisodeStatistics(self.BoundaryEnv('terminated')),
            lambda: self.gym.wrappers.RecordEpisodeStatistics(self.BoundaryEnv('continue')),
            lambda: self.gym.wrappers.RecordEpisodeStatistics(self.BoundaryEnv('both'))],
            autoreset_mode=self.gym.vector.AutoresetMode.SAME_STEP)
        try:
            obs, _ = envs.reset(seed=1)
            agent = PPO.Agent(1, 2, rngs=nnx.Rngs(default=jax.random.key(70)))
            storage = PPO.Storage(
                obs=jnp.zeros((1, 4, 1)), actions=jnp.zeros((1, 4), jnp.int32),
                logprobs=jnp.zeros((1, 4)), episode_ends=jnp.zeros((1, 4), jnp.bool_),
                values=jnp.zeros((1, 4)), advantages=jnp.zeros((1, 4)),
                returns=jnp.zeros((1, 4)), rewards=jnp.zeros((1, 4)))
            writer = mock.Mock()
            args = type('Args', (), {'num_steps': 1, 'num_envs': 4, 'gamma': 0.9})()
            # Compare the stored correction with the critic at the true final observation.
            _, final_value = agent(jnp.array([[10.]], jnp.float32))
            next_obs, storage, _, _ = PPO.rollout(agent, envs, obs, storage,
                jax.random.key(71), 0, writer, args)
            self.assertTrue(bool(jnp.allclose(storage.rewards[0, 0], 1 + 0.9 * final_value[0])))
            self.assertEqual(float(storage.rewards[0, 1]), 1.0)
            self.assertEqual(float(storage.rewards[0, 3]), 1.0)  # termination wins if both flags are true
            self.assertEqual(storage.episode_ends[0].tolist(), [True, True, False, True])
            self.assertEqual(next_obs[0, 0], 1.0)  # same-step reset observation
            self.assertEqual(writer.add_scalar.call_count, 6)  # return and length for each finished env
        finally:
            envs.close()


class RealEnvironmentSmokeTests(unittest.TestCase):
    def smoke(self, env_id, factory):
        import numpy as np
        env = factory()
        try:
            obs, _ = env.reset(seed=123)
            self.assertTrue(np.all(np.isfinite(obs)))
            for _ in range(4):
                obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
                self.assertTrue(np.all(np.isfinite(obs)), env_id)
                self.assertTrue(np.all(np.isfinite(reward)), env_id)
                if terminated or truncated:
                    obs, _ = env.reset()
        finally:
            env.close()

    def test_acrobot_factory_four_step_smoke(self):
        import PPO
        self.smoke('Acrobot-v1', PPO.make_env('Acrobot-v1', 0, False, 'smoke'))

    def test_pusher_factory_four_step_smoke(self):
        import importlib.util
        if importlib.util.find_spec('mujoco') is None:
            self.skipTest('core-only environment; install the mujoco extra')
        import PPO_continuous_action as continuous
        self.smoke('Pusher-v5', continuous.make_env('Pusher-v5', 0, False, 'smoke', 0.99))

    def test_atari_factory_four_step_smoke(self):
        import importlib.util
        if importlib.util.find_spec('ale_py') is None:
            self.skipTest('core-only environment; install the atari extra')
        import ale_py
        import gymnasium as gym
        import PPO_atari as atari
        gym.register_envs(ale_py)
        self.smoke('ALE/KungFuMaster-v5', atari.make_env('ALE/KungFuMaster-v5', 0, False, 'smoke'))


class MainCleanupTests(unittest.TestCase):
    def run_main_case(self, failure=None):
        import os
        import numpy as np
        import gymnasium as gym
        import jax
        import jax.numpy as jnp
        import PPO
        envs = mock.Mock()
        envs.single_action_space = gym.spaces.Discrete(2)
        envs.single_observation_space = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        envs.reset.return_value = (np.zeros((1, 3), np.float32), {})
        writer = mock.Mock()
        def collect(agent, envs, obs, storage, key, global_step, writer, args):
            if failure == 'interrupt': raise KeyboardInterrupt()
            if failure == 'train-error': raise RuntimeError('train fail')
            return obs, storage, key, global_step + 1
        rollout = mock.Mock(side_effect=collect)
        save = mock.Mock()
        if failure == 'save-error':
            save.side_effect = RuntimeError('save fail')
        writer_factory = mock.Mock(return_value=writer)
        if failure == 'startup-error':
            writer_factory.side_effect = RuntimeError('writer startup fail')
        original_cwd = os.getcwd()
        with TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with mock.patch.object(PPO.gym.vector, 'SyncVectorEnv', return_value=envs), \
                     mock.patch.object(PPO, 'SummaryWriter', writer_factory), \
                     mock.patch.object(PPO, 'rollout', rollout), \
                     mock.patch.object(PPO, 'compute_gae', return_value=(jnp.zeros((1, 1)), jnp.zeros((1, 1)))), \
                     mock.patch.object(PPO, 'update_ppo', return_value=(jnp.array(0.),
                         (jnp.array(0.),) * 4, jnp.array(0.), jax.random.key(1), jnp.array(3e-4))), \
                     mock.patch.object(PPO, 'tqdm', side_effect=lambda it: it), \
                     mock.patch('orbax.checkpoint.v1.save', save), \
                     mock.patch('sys.argv', ['PPO.py', '--num-envs', '1', '--num-steps', '1',
                         '--num-minibatches', '1', '--update-epochs', '1', '--total-timesteps', '1']):
                    if failure in ('train-error', 'save-error', 'startup-error'):
                        with self.assertRaisesRegex(RuntimeError, 'train fail|save fail|writer startup fail'):
                            PPO.main()
                    else:
                        PPO.main()
            finally:
                os.chdir(original_cwd)
        envs.close.assert_called_once()
        if failure == 'startup-error':
            writer.close.assert_not_called()
            save.assert_not_called()
        else:
            writer.close.assert_called_once()
        self.assertEqual(save.call_count, 0 if failure in ('train-error', 'startup-error') else 1)

    def test_cleanup_on_normal_run_and_handled_interrupt(self):
        self.run_main_case()
        self.run_main_case('interrupt')

    def test_cleanup_on_training_startup_and_save_failures(self):
        self.run_main_case('train-error')
        self.run_main_case('startup-error')
        self.run_main_case('save-error')

    def test_continuous_and_atari_mains_close_resources_on_training_error(self):
        import os
        import importlib
        import numpy as np
        import gymnasium as gym
        import jax
        import PPO_continuous_action as continuous
        import PPO_atari as atari
        for module, action_space, obs_shape in (
            (continuous, gym.spaces.Box(-1, 1, (1,), dtype=np.float32), (3,)),
            (atari, gym.spaces.Discrete(3), (84, 84, 1)),
        ):
            envs = mock.Mock()
            envs.single_action_space = action_space
            envs.single_observation_space = gym.spaces.Box(-1, 1, obs_shape, dtype=np.float32)
            envs.reset.return_value = (np.zeros((1,) + obs_shape, np.float32), {})
            writer = mock.Mock()
            agent = object()
            optimizer = object()
            original_cwd = os.getcwd()
            with TemporaryDirectory() as directory:
                try:
                    os.chdir(directory)
                    with mock.patch.object(module.gym.vector, 'SyncVectorEnv', return_value=envs), \
                         mock.patch.object(module, 'Agent', return_value=agent), \
                         mock.patch.object(module.nnx, 'Optimizer', return_value=optimizer), \
                         mock.patch.object(module.nnx, 'state', return_value={}), \
                         mock.patch.object(module, 'SummaryWriter', return_value=writer), \
                         mock.patch.object(module, 'rollout', side_effect=RuntimeError('train fail')), \
                         mock.patch.object(module, 'tqdm', side_effect=lambda it: it), \
                         mock.patch('orbax.checkpoint.v1.save') as save, \
                         mock.patch('sys.argv', [module.__name__ + '.py', '--num-envs', '1',
                             '--num-steps', '1', '--num-minibatches', '1', '--update-epochs', '1',
                             '--total-timesteps', '1']):
                        with self.assertRaisesRegex(RuntimeError, 'train fail'):
                            module.main()
                finally:
                    os.chdir(original_cwd)
            envs.close.assert_called_once()
            writer.close.assert_called_once()
            save.assert_not_called()

    def test_continuous_wrappers_clip_observations_rewards_and_delivered_actions(self):
        import gymnasium as gym
        import numpy as np
        import jax
        import jax.numpy as jnp
        import PPO_continuous_action as continuous
        class Env(gym.Env):
            observation_space = gym.spaces.Box(-1000, 1000, shape=(2,), dtype=np.float32)
            action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
            def __init__(self): self.received_actions = []
            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(2, np.float32), {}
            def step(self, action):
                self.received_actions.append(np.array(action))
                return np.array([1000., -1000.], np.float32), 2.0, True, False, {}
        raw = Env()
        with mock.patch.object(continuous.gym, 'make', return_value=raw):
            wrapped = continuous.make_env('TestContinuous-v0', 0, False, 'run', 0.9)()
        envs = gym.vector.SyncVectorEnv([lambda: wrapped],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
        try:
            obs, _ = envs.reset(seed=3)
            self.assertEqual(obs.dtype, np.float32)
            self.assertTrue(np.all(obs >= -10) and np.all(obs <= 10))
            storage = continuous.Storage(
                obs=jnp.zeros((1, 1, 2)), actions=jnp.zeros((1, 1, 1)), logprobs=jnp.zeros((1, 1)),
                episode_ends=jnp.zeros((1, 1), jnp.bool_), values=jnp.zeros((1, 1)),
                advantages=jnp.zeros((1, 1)), returns=jnp.zeros((1, 1)), rewards=jnp.zeros((1, 1)))
            args = type('Args', (), {'num_steps': 1, 'num_envs': 1, 'gamma': 0.99})()
            writer = mock.Mock()
            with mock.patch.object(continuous, 'get_action_and_value', return_value=(
                    jnp.array([[5.]], jnp.float32), jnp.array([0.]), jnp.array([0.]), jax.random.key(5))):
                _, storage, _, _ = continuous.rollout(None, envs, obs, storage,
                    jax.random.key(4), 0, writer, args)
            self.assertEqual(float(storage.actions[0, 0, 0]), 5.0)
            self.assertEqual(float(raw.received_actions[0][0]), 1.0)
            self.assertEqual(float(writer.add_scalar.call_args_list[0].args[1]), 2.0)
            clipped_obs, _, _, _, _ = envs.step(np.array([[0.]], np.float32))
            self.assertEqual(clipped_obs.dtype, np.float32)
            self.assertTrue(np.all(clipped_obs >= -10) and np.all(clipped_obs <= 10))
        finally:
            envs.close()


if __name__ == '__main__':
    unittest.main()
