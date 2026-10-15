import numpy as np
import torch
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.agents.baselines.gp_greedy import GPGreedy
from mars_ipp.agents.baselines.gp_trace import GPTrace
from mars_ipp.agents.baselines.gp_entropy import GPEntropy
from mars_ipp.agents.baselines.coverage import Coverage
from mars_ipp.agents.baselines.msr import Msr
from mars_ipp.envs.grid_env import GridEnv

BASELINES = {'gp_greedy': GPGreedy, 'gp_trace': GPTrace, 'gp_entropy': GPEntropy, 'cover': Coverage, 'msr': Msr}


def env_config(config):
    return {**config['env'], 'reward': config['reward'],
            'observation_mode': config['ppo'].get('observation_mode', 'full')}


def metric_row(info, seed, episode, algorithm):
    row = {key: value for key, value in info.items() if key not in ('raw_observation', 'position')}
    row.update({'seed': seed, 'episode': episode, 'algorithm': algorithm})
    return row


def evaluate(field, scaler, config, agent=None, algorithm='ppo', episode=0, metrics_mode='full', action_sampler=None):
    rows, trajectories = [], []
    snapshot = None
    if algorithm in ('ppo', 'msr', 'ours') and agent is None:
        raise ValueError('Msr evaluation requires a trained policy')
    random_cover = config['evaluation'].get('cover_random_start', False)
    if hasattr(scaler, 'for_scene'):
        scaler = scaler.for_scene(config['field']['test_scenario'])
    original_scaler = agent.scaler if agent is not None else None
    seeds = config['evaluation']['seeds'][:1] if algorithm == 'cover' and not random_cover else config['evaluation']['seeds']
    for seed in seeds:
        env = GridEnv(field, scaler, env_config(config), metrics_mode=metrics_mode)
        policy = agent if algorithm in ('ppo', 'msr', 'ours') else BASELINES[algorithm](seed=seed)
        if agent is not None:
            agent.scaler = scaler
        if algorithm == 'cover' and random_cover:
            policy = Coverage(seed=seed, random_start=True, size=config['env']['grid_size'])
        generator = torch.Generator(device=getattr(policy, 'device', 'cpu')).manual_seed(seed + 100000)
        observation, info = env.reset(seed=seed, options={'start': Coverage.start} if algorithm == 'cover' and not random_cover else None)
        if metrics_mode == 'full':
            rows.append(metric_row(env.prior_info, seed, episode, algorithm))
            rows.append(metric_row(info, seed, episode, algorithm))
        path = [env.position.copy()] if metrics_mode == 'full' else None
        episode_reward = 0.
        while not env.finished:
            if action_sampler is not None:
                action = action_sampler(policy, observation, generator)
            else:
                action = policy.act(observation, generator=generator)[0] if isinstance(policy, PPO) else policy.select_action(observation, env.planning_context())
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if metrics_mode == 'full':
                path.append(env.position.copy())
            if metrics_mode == 'full' or env.finished:
                rows.append(metric_row(info, seed, episode, algorithm))
            if metrics_mode == 'full':
                trajectories.append({'episode': episode, 'seed': seed, 'algorithm': algorithm, 'step': info['step'],
                                     'i': int(env.position[0]), 'j': int(env.position[1]), 'action': action, 'reward': reward,
                                     **{name: float(env.measurement[i]) for i, name in enumerate(('Br', 'B2', 'B3'))}})
        if metrics_mode == 'terminal' and config['evaluation'].get('record_episode_reward', False):
            rows[-1]['episode_reward'] = episode_reward
        if metrics_mode == 'full' and snapshot is None:
            mean, variance = env.gp.predict(env.evaluation_points)
            snapshot = {'points': env.evaluation_points, 'reference': env.reference, 'mean': mean,
                        'variance': variance, 'trajectory': np.asarray(path) * env.scale}
    if agent is not None:
        agent.scaler = original_scaler
    return rows, trajectories, snapshot
