# go1_multiobject_wrapper.py

import torch
from gym import spaces
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

class Go1MultiObjectWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.num_npcs = getattr(self.cfg.env, "num_npcs", 0)

        # Adjust observation space based on the number of NPCs
        obs_dim = 2 + 3 * self.num_agents + 3 * self.num_npcs  # Adjusted
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(obs_dim,), dtype=float)

        # Action space remains the same unless agents can interact with NPCs
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[0.5, 0.5, 0.5],]], device=self.device).repeat(self.num_envs, self.num_agents, 1)

        # Initialize reward scales (update as necessary)
        self.approach_reward_scale = self.cfg.rewards.scales.approach_reward_scale
        self.target_reward_scale = self.cfg.rewards.scales.target_reward_scale
        self.reach_target_reward_scale = self.cfg.rewards.scales.reach_target_reward_scale
        self.collision_punishment_scale = self.cfg.rewards.scales.collision_punishment_scale
        self.push_reward_scale = self.cfg.rewards.scales.push_reward_scale
        self.ocb_reward_scale = self.cfg.rewards.scales.ocb_reward_scale
        self.exception_punishment_scale = self.cfg.rewards.scales.exception_punishment_scale

        # Initialize reward buffer for multiple NPCs
        self.reward_buffer = {
            "distance_to_target_reward": 0,
            "exception_punishment": 0,
            "approach_to_npc_reward": torch.zeros(self.num_npcs, device=self.device),  # Adjusted
            "collision_punishment": torch.zeros(self.num_npcs, device=self.device),   # Adjusted
            "reach_target_reward": 0,
            "push_reward": 0,
            "ocb_reward": 0,
            "step_count": 0,
        }

    def _init_extras(self, obs):
        pass

    def _get_obs(self, observations):
        # Extract observations and concatenate NPC observations
        agent_obs = observations['agents']  # Shape: [num_envs, num_agents, obs_dim]
        npc_obs = observations['npcs']      # Shape: [num_envs, num_npcs, obs_dim]

        # Flatten observations
        agent_obs_flat = agent_obs.view(self.num_envs, -1)
        npc_obs_flat = npc_obs.view(self.num_envs, -1)

        # Combine agent and NPC observations
        obs = torch.cat([agent_obs_flat, npc_obs_flat], dim=1)
        return obs

    def _compute_reward(self, observations, actions, done):
        # Compute rewards based on distances to NPCs, collisions, etc.
        approach_rewards = torch.zeros(self.num_envs, device=self.device)
        collision_punishments = torch.zeros(self.num_envs, device=self.device)

        for i in range(self.num_npcs):
            # Compute distance between agent and NPC[i]
            npc_pos = observations['npcs'][:, i, :3]  # NPC positions
            agent_pos = observations['agents'][:, 0, :3]  # Assuming single agent
            distance = torch.norm(agent_pos - npc_pos, dim=1)

            # Update rewards
            approach_rewards += self.approach_reward_scale / (distance + 1e-6)
            # Check for collisions and update punishments
            collision = (distance < self.collision_threshold).float()
            collision_punishments -= collision * self.collision_punishment_scale

        total_reward = approach_rewards + collision_punishments
        self.reward_buffer['approach_to_npc_reward'] = approach_rewards
        self.reward_buffer['collision_punishment'] = collision_punishments

        return total_reward

    def step(self, actions):
        observations, rewards, dones, infos = self.env.step(actions)

        # Get custom observations
        obs = self._get_obs(observations)

        # Compute custom rewards
        rewards = self._compute_reward(observations, actions, dones)

        # Update infos if needed
        infos['reward_buffer'] = self.reward_buffer.copy()

        return obs, rewards, dones, infos