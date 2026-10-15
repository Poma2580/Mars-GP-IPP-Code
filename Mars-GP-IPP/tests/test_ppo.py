import numpy as np
import pytest
import torch
from mars_ipp.agents.ppo.buffer import compute_gae
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.training.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint
from mars_ipp.utils.seed import seed_all


def test_gae_terminal_and_unscaled_target():
    adv, targets = compute_gae([2], [.5], [99], [True], [True], .99, .9)
    np.testing.assert_allclose(adv, [1.5])
    np.testing.assert_allclose(targets, [2])


def test_gae_multiple_episodes_and_rollout_bootstrap():
    adv, targets = compute_gae([1, 2, 3], [.1, .2, .3], [.2, 100, .4], [False, True, False], [False, True, False], .9, .8)
    np.testing.assert_allclose(adv, [1+.9*.2-.1+.9*.8*1.8, 1.8, 3+.9*.4-.3])
    np.testing.assert_allclose(targets, adv + [.1,.2,.3])
    # External truncation bootstraps but cannot link the next reset episode.
    adv, _ = compute_gae([1,99], [0,0], [2,0], [False,True], [True,True], .9,.8)
    assert adv[0] == pytest.approx(2.8)


def test_one_sample_update_and_inference_checkpoint(config, scaler, tmp_path):
    seed_all(4)
    agent = PPO(config['ppo'], scaler)
    obs = np.array([0., 0., 1., 2., 3.])
    action, logprob, value, state = agent.act(obs)
    agent.buffer.add(state, action, logprob, value, 1., 0., True, True)
    original_parameters = [p.clone() for p in agent.network.parameters()]
    report = agent.update()
    assert np.isfinite(report['loss']) and len(agent.buffer) == 0
    assert any(not torch.equal(a, b) for a, b in zip(original_parameters, agent.network.parameters()))
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, agent, config, {'episode': 1, 'steps': 1})
    payload = read_checkpoint(path)
    assert set(payload) == {'format', 'model_version', 'network', 'config', 'scaler', 'progress', 'algorithm', 'input_dim'}
    assert payload['format'] == 2
    restored = PPO(config['ppo'], scaler)
    restore_checkpoint(payload, restored)
    assert len(restored.buffer) == 0 and restored.updates == 0
    np.testing.assert_allclose(agent.network.actor(torch.as_tensor(state, device=agent.device)).detach().cpu(),
                               restored.network.actor(torch.as_tensor(state, device=restored.device)).detach().cpu())
    for a, b in zip(agent.network.parameters(), restored.network.parameters()):
        assert torch.equal(a, b)
    # Old inference loading must not need or restore continuation state.
    old = {**payload, 'format': 1, 'optimizer': {}, 'buffer': ['ignored'],
           'rng': {}, 'env_rng': {}, 'history': {}, 'updates': 99}
    old.pop('model_version')
    torch.save(old, tmp_path / 'format1.pt')
    restore_checkpoint(read_checkpoint(tmp_path / 'format1.pt'), restored)
    assert len(restored.buffer) == 0 and restored.updates == 0
    invalid = agent.network.state_dict()
    invalid['actor.0.weight'] = torch.zeros(64, 2)
    with pytest.raises(ValueError, match='input dimension'):
        agent.load_network(invalid)
