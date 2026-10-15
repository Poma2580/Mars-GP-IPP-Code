"""SB3 bridge for the finite-horizon V3 experiment (no normalization wrapper)."""
from contextlib import contextmanager
from pathlib import Path
import importlib.metadata
import hashlib
import random
import time
import numpy as np
import torch
import yaml
import gymnasium as gym
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from mars_ipp.envs.grid_env import GridEnv
from mars_ipp.envs.observations import make_scaler
from mars_ipp.evaluation.evaluator import evaluate, env_config
from mars_ipp.field.generate_field import prepare_fields
from mars_ipp.utils.seed import seed_all
from mars_ipp.utils.logger import append_csv, write_json


@contextmanager
def isolated_rng():
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        yield
    finally:
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])


class GymGridEnv(gym.Env):
    metadata = {}

    def __init__(self, field, scaler, config):
        self.core = GridEnv(field, scaler, env_config(config), metrics_mode='terminal')
        self.scaler = scaler
        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Box(0., 1., (11,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observation, info = self.core.reset(seed=seed, options=options)
        return self.scaler.transform(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.core.step(int(action))
        return self.scaler.transform(observation), float(reward), bool(terminated), bool(truncated), info


class SB3Inference:
    def __init__(self, model, scaler):
        self.model, self.scaler, self.device = model, scaler, model.device

    @torch.no_grad()
    def probabilities(self, observation):
        tensor = torch.as_tensor(self.scaler.transform(observation), device=self.device).unsqueeze(0)
        return self.model.policy.get_distribution(tensor).distribution.probs[0]

    @torch.no_grad()
    def value(self, observation):
        tensor = torch.as_tensor(self.scaler.transform(observation), device=self.device).unsqueeze(0)
        return self.model.policy.predict_values(tensor).item()

    def sample(self, observation, generator):
        return int(torch.multinomial(self.probabilities(observation), 1, generator=generator).item())


def sample_sb3(policy, observation, generator):
    return policy.sample(observation, generator)


def make_model(env, config):
    p = config['ppo']; s = config['sb3']
    return SB3PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=p['rollout_steps'],
                  batch_size=s['batch_size'], n_epochs=p['update_epochs'], gamma=.99,
                  gae_lambda=.9, clip_range=.2, clip_range_vf=None, ent_coef=.01,
                  vf_coef=.5, max_grad_norm=.5, normalize_advantage=True, target_kl=None,
                  policy_kwargs={'net_arch': {'pi': [64,64], 'vf': [64,64]}, 'activation_fn': torch.nn.Tanh},
                  device=p['device'], seed=config['seed'], verbose=0)


class ExperimentCallback(BaseCallback):
    """Read diagnostics and persist only AFTER train(), including the last rollout."""
    def __init__(self, config, directory, field, scaler):
        super().__init__()
        self.config, self.directory, self.field, self.scaler = config, directory, field, scaler
        self.episodes = 0; self.reward_sum = 0.; self.last_update = 0; self.started = time.monotonic()
        self.total = config['training']['episodes'] * config['env']['max_steps']

    def _on_step(self):
        self.reward_sum += float(self.locals['rewards'][0])
        if self.locals['dones'][0]:
            self.episodes += 1
            info = self.locals['infos'][0]
            append_csv(self.directory/'train_metrics.csv', [{'episode': self.episodes, 'steps': self.num_timesteps,
                'reward': self.reward_sum, 'r_gp': self.reward_sum,
                **{k: info[k] for k in ('rmse','trace_current','trace_prior','trr','unique_positions','repeated_visits','outside_actions')},
                'mean_reward': self.reward_sum/self.config['env']['max_steps']}])
            self.reward_sum = 0.
        return True

    def _on_rollout_start(self):
        self.after_update()

    def _on_training_end(self):
        self.after_update()

    def after_update(self):
        if self.num_timesteps == self.last_update:
            return
        self.last_update = self.num_timesteps
        values = self.model.logger.name_to_value
        report = {'steps': self.num_timesteps, 'update': self.num_timesteps//self.model.n_steps,
                  'gradient_updates': self.model._n_updates*(self.model.n_steps//self.model.batch_size)}
        for key in ('approx_kl','clip_fraction','entropy_loss','value_loss','explained_variance','loss'):
            v = float(values.get('train/'+key, np.nan))
            report[key] = v if np.isfinite(v) else None
        append_csv(self.directory/'updates.csv', [report])
        if self.episodes % self.config['training']['eval_every'] == 0 or self.num_timesteps == self.total:
            cfg = {**self.config, 'evaluation': {**self.config['evaluation'], 'seeds': self.config['training']['monitor_seeds']}}
            with isolated_rng():
                rows, _, _ = evaluate(self.field, self.scaler, cfg, SB3Inference(self.model,self.scaler),
                                      episode=self.episodes, metrics_mode='terminal', action_sampler=sample_sb3)
            append_csv(self.directory/'eval_endpoints.csv', rows)
            self.model.save(self.directory/'checkpoint.zip')
            write_json(self.directory/'run_status.json', {'episode': self.episodes,'steps': self.num_timesteps,
                'completed': self.num_timesteps == self.total, 'updates': report['update'],
                'gradient_updates': report['gradient_updates'], 'pending_rollout': 0,
                'invocation_seconds': time.monotonic()-self.started})
            print(f"sb3 episode={self.episodes} steps={self.num_timesteps} eval_rmse={np.mean([r['rmse'] for r in rows]):.4f}", flush=True)


def train_sb3(config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    seed_all(config['seed'])
    fields = prepare_fields(config); scaler = make_scaler(fields, config)
    field = fields['train1']
    env = DummyVecEnv([lambda: GymGridEnv(field, scaler, config)])
    model = make_model(env, config)
    (directory/'config.yaml').write_text(yaml.safe_dump(config))
    write_json(directory/'metadata.json', {'scaler': scaler.to_dict(), 'fields': {
        name: {'source': f.source,'sha256': f.sha256,'parameters': f.parameters} for name,f in fields.items()},
        'versions': {n: importlib.metadata.version(n) for n in ('stable-baselines3','gymnasium','cloudpickle','farama-notifications','torch','numpy','scipy','GPy')},
        'model_version':'sb3-ppo-11d-v3', 'config':config,
        'source_sha256': {str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parents[1].rglob('*.py'))}})
    cb = ExperimentCallback(config,directory,field,scaler)
    model.learn(total_timesteps=config['training']['episodes']*70, callback=cb)
    restored = SB3PPO.load(directory/'checkpoint.zip', device=config['ppo']['device'])
    probe, _ = GymGridEnv(field,scaler,config).reset(seed=1000)
    with torch.no_grad():
        x = torch.as_tensor(probe,device=model.device).unsqueeze(0)
        assert torch.equal(model.policy.get_distribution(x).distribution.probs, restored.policy.get_distribution(x).distribution.probs)
    env.close()
    return directory
