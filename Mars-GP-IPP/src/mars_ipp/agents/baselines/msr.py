from mars_ipp.agents.ppo.ppo import PPO


class Msr(PPO):
    """Independent coordinate-only PPO; its reward is provided by GridEnv."""
    def __init__(self, config, scaler):
        if config.get('observation_mode') != 'coordinates_only':
            raise ValueError('Msr requires a genuine 2D coordinates_only policy')
        super().__init__(config, scaler)
