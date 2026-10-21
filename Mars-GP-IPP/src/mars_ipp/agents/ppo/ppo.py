import numpy as np
import torch
from mars_ipp.agents.ppo.masking import action_mask
from .network import ActorCritic
from .buffer import RolloutBuffer, compute_gae


class PPO:
    def __init__(self, config, scaler):
        self.config, self.scaler = dict(config), scaler
        self.observation_mode = config.get('observation_mode', 'full')
        if self.observation_mode not in ('full', 'position_only', 'coordinates_only', 'full_bnorm', 'full_uncertainty', 'full_uncertainty_local_global_3x3', 'gp_trace_local_11d', 'gp_trace_b_14d'):
            raise ValueError('observation_mode must be full or position_only')
        self.device = torch.device(config.get('device', 'cuda'))
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; set ppo.device=cpu explicitly for a CPU run')
        self.input_dim = {'gp_trace_b_14d': 14, 'gp_trace_local_11d': 11, 'coordinates_only': 2, 'full_uncertainty': 13, 'full_uncertainty_local_global_3x3': 22}.get(self.observation_mode, 5)
        self.network = ActorCritic(self.input_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.network.actor.parameters(), 'lr': config['actor_lr']},
            {'params': self.network.critic.parameters(), 'lr': config['critic_lr']}])
        self.buffer = RolloutBuffer()
        self.updates = 0
        self.action_mask_enabled = config.get("action_mask", False)

    def prepare_state(self, observation):
        """Mask policy features only, after training-field normalization."""
        if self.observation_mode == 'coordinates_only':
            return (np.asarray(observation[:2]) / self.scaler.coordinate_scale).astype(np.float32)
        state = self.scaler.transform(observation)
        if self.observation_mode == 'position_only':
            state[2:] = 0.
        return state

    def mask_for(self, observation):
        return action_mask(np.asarray(observation[:2]), int(self.scaler.coordinate_scale) + 1) if self.action_mask_enabled else None

    @torch.no_grad()
    def act(self, observation, generator=None):
        state = self.prepare_state(observation)
        tensor = torch.as_tensor(state, device=self.device)
        mask = self.mask_for(observation)
        probabilities = (self.network.distribution(tensor, mask).probs if mask is not None
                         else self.network.actor(tensor))
        action = int(torch.multinomial(probabilities, 1, generator=generator).item())
        logprob = torch.distributions.Categorical(probabilities).log_prob(torch.tensor(action, device=self.device)).item()
        value = self.network.critic(tensor).item()
        return action, logprob, value, state

    def select_action(self, observation, planning_context=None):
        return self.act(observation)[0]

    @torch.no_grad()
    def value(self, observation):
        return self.network.critic(torch.as_tensor(self.prepare_state(observation), device=self.device)).item()

    def update(self):
        if not self.buffer.transitions:
            return None
        masks = None
        if self.action_mask_enabled:
            if any(len(t) != 9 for t in self.buffer.transitions):
                raise ValueError('Masked PPO requires historical rollout masks')
            masks = torch.as_tensor(np.asarray([t[8] for t in self.buffer.transitions]), device=self.device)
        states, actions, logprobs, values, rewards, next_values, terminated, ends = zip(*(t[:8] for t in self.buffer.transitions))
        adv, targets = compute_gae(rewards, values, next_values, terminated, ends, self.config['gamma'], self.config['gae_lambda'])
        states = torch.tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(logprobs, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(adv, dtype=torch.float32, device=self.device)
        targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        if self.config['advantage_normalization']:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        for _ in range(self.config['update_epochs']):
            logprob, predicted, entropy = self.network.evaluate(states, actions, masks)
            ratio = torch.exp(logprob - old_logprobs)
            if self.action_mask_enabled:
                if (not masks.gather(1, actions[:, None]).all()
                        or not all(torch.isfinite(x).all() for x in (logprob, old_logprobs, entropy, ratio))):
                    raise FloatingPointError('Invalid masked PPO probabilities or importance ratio')
            policy_loss = -torch.minimum(ratio * advantages, ratio.clamp(1-self.config['clip'], 1+self.config['clip']) * advantages).mean()
            value_loss = torch.mean((predicted - targets) ** 2)
            loss = policy_loss + self.config['value_coef'] * value_loss - self.config['entropy_coef'] * entropy.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite PPO loss')
            self.optimizer.zero_grad()
            loss.backward()
            if self.action_mask_enabled and any(p.grad is not None and not torch.isfinite(p.grad).all()
                                                for p in self.network.parameters()):
                raise FloatingPointError('Nonfinite masked PPO gradient')
            self.optimizer.step()
        self.updates += 1
        report = {'update': self.updates, 'samples': len(self.buffer), 'loss': loss.item(),
                  'policy_loss': policy_loss.item(), 'value_loss': value_loss.item(), 'entropy': entropy.mean().item()}
        if self.config.get('readonly_diagnostics', False):
            with torch.no_grad():
                lp, vp, ent = self.network.evaluate(states, actions, masks)
                log_ratio = lp - old_logprobs
                variance = torch.var(targets, unbiased=False).item()
                report.update(approx_kl=(torch.exp(log_ratio)-1-log_ratio).mean().item(),
                              clip_fraction=((torch.exp(log_ratio)-1).abs() > self.config['clip']).float().mean().item(),
                              explained_variance=1-torch.var(targets-vp, unbiased=False).item()/variance if variance > 0 else None,
                              gradient_updates=self.updates*self.config['update_epochs'])
        self.buffer.clear()
        return report

    def load_network(self, state):
        if any(f'{name}.0.weight' not in state or state[f'{name}.0.weight'].shape[1] != self.input_dim for name in ('actor', 'critic')):
            raise ValueError('Checkpoint input dimension differs from policy')
        self.network.load_state_dict(state)
