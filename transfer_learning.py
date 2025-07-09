import asyncio
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Simple environment simulator (replacing gym for Pyodide compatibility)
class SimpleEnv:
    def __init__(self, is_atari=True):
        self.is_atari = is_atari
        if is_atari:
            self.action_space = SimpleActionSpace(n=6)  # Mimic Pong actions
            self.observation_space = SimpleObservationSpace(shape=(4, 84, 84))
        else:
            self.action_space = SimpleActionSpace(shape=(2,), high=np.array([1.0, 1.0]))
            self.observation_space = SimpleObservationSpace(shape=(8,))
        self.state = self.reset()

    def reset(self):
        if self.is_atari:
            return np.random.rand(4, 84, 84)
        return np.random.rand(8)

    def step(self, action):
        next_state = self.reset()
        reward = np.random.rand() * 2 - 1
        done = np.random.random() < 0.1
        return next_state, reward, done, {}

class SimpleActionSpace:
    def __init__(self, n=None, shape=None, high=None):
        self.n = n
        self.shape = shape
        self.high = high

    def sample(self):
        if self.n:
            return np.random.randint(self.n)
        return np.random.uniform(-self.high, self.high, self.shape)

class SimpleObservationSpace:
    def __init__(self, shape):
        self.shape = shape

# GATN Architecture
class GATN(nn.Module):
    def __init__(self, input_shape, action_dim, max_action=None, num_source_tasks=1, is_atari=True):
        super(GATN, self).__init__()
        self.is_atari = is_atari
        self.max_action = max_action if max_action is not None else 1.0
        self.action_dim = action_dim
        self.num_source_tasks = num_source_tasks

        if is_atari:
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(7 * 7 * 64, 512),
                nn.ReLU()
            )
            self.feature_dim = 512
        else:
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_shape, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            self.feature_dim = 64

        if is_atari:
            self.policy_head = nn.Linear(self.feature_dim, action_dim)
        else:
            self.policy_head = nn.Sequential(
                nn.Linear(self.feature_dim, action_dim),
                nn.Tanh()
            )

        if not is_atari:
            self.critic = nn.Sequential(
                nn.Linear(self.feature_dim + action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )

        self.transfer_attention = nn.Sequential(
            nn.Linear(self.feature_dim, num_source_tasks),
            nn.Softmax(dim=-1)
        )

    def forward_actor(self, state, source_policies=None):
        features = self.feature_extractor(state)
        action = self.policy_head(features)
        if not self.is_atari:
            action = action * self.max_action
        if source_policies is not None:
            attention_weights = self.transfer_attention(features)
            transferred_action = torch.zeros_like(action)
            for i in range(self.num_source_tasks):
                transferred_action += attention_weights[:, i].unsqueeze(1) * source_policies[i].forward_actor(state)
            action = action + transferred_action
        return action

    def forward_critic(self, state, action):
        if self.is_atari:
            raise ValueError("Critic not used for Atari (DQN-based)")
        features = self.feature_extractor(state)
        critic_input = torch.cat([features, action], dim=-1)
        return self.critic(critic_input)

# GATN Agent
class GATNAgent:
    def __init__(self, env, source_policies=None, is_atari=True):
        self.env = env
        self.device = torch.device("cpu")
        self.is_atari = is_atari
        if is_atari:
            self.input_shape = (4, 84, 84)
            self.action_dim = env.action_space.n
            self.max_action = None
        else:
            self.input_shape = env.observation_space.shape[0]
            self.action_dim = env.action_space.shape[0]
            self.max_action = float(env.action_space.high[0])

        self.model = GATN(self.input_shape, self.action_dim, self.max_action, len(source_policies) if source_policies else 0, is_atari).to(self.device)
        self.target_model = GATN(self.input_shape, self.action_dim, self.max_action, len(source_policies) if source_policies else 0, is_atari).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.memory = deque(maxlen=10000)
        self.source_policies = source_policies if source_policies else []
        self.gamma = 0.99
        self.batch_size = 64 if not is_atari else 32
        self.steps = 0
        self.tau = 0.005 if not is_atari else None
        self.noise_scale = 0.1 if not is_atari else None

        if is_atari:
            self.epsilon = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.target_update_freq = 1000
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        else:
            self.actor_optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            self.critic_optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def preprocess_frame(self, state):
        if self.is_atari:
            return torch.tensor(state, dtype=torch.float32).to(self.device) / 255.0
        return torch.tensor(state, dtype=torch.float32).to(self.device)

    def act(self, state, add_noise=True):
        state = self.preprocess_frame(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.is_atari:
                if random.random() < self.epsilon:
                    return self.env.action_space.sample()
                q_values = self.model.forward_actor(state, self.source_policies)
                return q_values.argmax().item()
            else:
                action = self.model.forward_actor(state, self.source_policies).cpu().numpy()[0]
                if add_noise:
                    noise = np.random.normal(0, self.noise_scale * self.max_action, size=self.action_dim)
                    action = (action + noise).clip(-self.max_action, self.max_action)
                return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack([self.preprocess_frame(s) for s in states]).to(self.device)
        next_states = torch.stack([self.preprocess_frame(s) for s in next_states]).to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        dones = torch.from_numpy(np.array(dones)).float().to(self.device)

        if self.is_atari:
            actions = torch.tensor(actions, dtype=torch.long).to(self.device)
            current_q = self.model.forward_actor(states, self.source_policies).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q = self.target_model.forward_actor(next_states, self.source_policies).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
            loss = nn.MSELoss()(current_q, target_q.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.steps += 1
            if self.steps % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
        else:
            actions = torch.from_numpy(np.array(actions)).float().to(self.device)
            target_actions = self.target_model.forward_actor(next_states, self.source_policies)
            target_q = self.target_model.forward_critic(next_states, target_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q.squeeze()
            current_q = self.model.forward_critic(states, actions).squeeze()
            critic_loss = nn.MSELoss()(current_q, target_q.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            actions_pred = self.model.forward_actor(states, self.source_policies)
            actor_loss = -self.model.forward_critic(states, actions_pred).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            self.steps += 1

# Main loop
async def main():
    atari_env = SimpleEnv(is_atari=True)
    mujoco_env = SimpleEnv(is_atari=False)

    atari_source_policies = [
        GATN((4, 84, 84), atari_env.action_space.n, is_atari=True).to(torch.device("cpu")),
        GATN((4, 84, 84), atari_env.action_space.n, is_atari=True).to(torch.device("cpu"))
    ]
    mujoco_source_policies = [
        GATN(mujoco_env.observation_space.shape[0], mujoco_env.action_space.shape[0], float(mujoco_env.action_space.high[0]), is_atari=False).to(torch.device("cpu"))
    ]

    atari_agent = GATNAgent(atari_env, atari_source_policies, is_atari=True)
    mujoco_agent = GATNAgent(mujoco_env, mujoco_source_policies, is_atari=False)

    episodes = 10
    for episode in range(episodes):
        # Atari
        state = atari_env.reset()
        total_reward_atari = 0
        done = False
        while not done:
            action = atari_agent.act(state)
            next_state, reward, done, _ = atari_env.step(action)
            atari_agent.remember(state, action, reward, next_state, done)
            atari_agent.replay()
            state = next_state
            total_reward_atari += reward
            await asyncio.sleep(1.0 / 60)
        print(f"Atari Episode {episode + 1}, Total Reward: {total_reward_atari}")

        # MuJoCo
        state = mujoco_env.reset()
        total_reward_mujoco = 0
        done = False
        while not done:
            action = mujoco_agent.act(state)
            next_state, reward, done, _ = mujoco_env.step(action)
            mujoco_agent.remember(state, action, reward, next_state, done)
            mujoco_agent.replay()
            state = next_state
            total_reward_mujoco += reward
            await asyncio.sleep(1.0 / 60)
        print(f"MuJoCo Episode {episode + 1}, Total Reward: {total_reward_mujoco}")

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
