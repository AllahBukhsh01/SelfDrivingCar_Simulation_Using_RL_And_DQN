# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import random
# from collections import deque

# # =============================
# # DQN Neural Network
# # =============================
# class DQN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(DQN, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, output_dim)
#         )

#     def forward(self, x):
#         return self.fc(x)


# # =============================
# # Replay Buffer
# # =============================
# class ReplayBuffer:
#     def __init__(self, capacity=10000):
#         self.buffer = deque(maxlen=capacity)

#     def push(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))

#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
#         return states, actions, rewards, next_states, dones

#     def __len__(self):
#         return len(self.buffer)


# # =============================
# # DQN Agent
# # =============================
# class DQNAgent:
#     def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = DQN(state_dim, action_dim).to(self.device)
#         self.target_model = DQN(state_dim, action_dim).to(self.device)
#         self.target_model.load_state_dict(self.model.state_dict())
#         self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
#         self.gamma = gamma
#         self.memory = ReplayBuffer()
#         self.action_dim = action_dim
#         self.epsilon = 1.0
#         self.epsilon_min = 0.05
#         self.epsilon_decay = 0.995

#     def act(self, state):
#         if np.random.rand() < self.epsilon:
#             return np.random.randint(self.action_dim)
#         state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             q_values = self.model(state)
#         return q_values.argmax().item()

#     def update(self, batch_size=64):
#         if len(self.memory) < batch_size:
#             return

#         states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

#         states = torch.FloatTensor(states).to(self.device)
#         actions = torch.LongTensor(actions).to(self.device)
#         rewards = torch.FloatTensor(rewards).to(self.device)
#         next_states = torch.FloatTensor(next_states).to(self.device)
#         dones = torch.FloatTensor(dones).to(self.device)

#         # Compute target Q-values
#         with torch.no_grad():
#             max_next_q = self.target_model(next_states).max(1)[0]
#             target_q = rewards + (1 - dones) * self.gamma * max_next_q

#         # Compute current Q-values
#         current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

#         # Loss
#         loss = nn.MSELoss()(current_q, target_q)

#         # Optimize
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#     def update_target(self):
#         self.target_model.load_state_dict(self.model.state_dict())


#=================================================================================================

#                                     REAL TRACK VERSION

#=================================================================================================


# # agent.py
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import random
# from collections import deque

# class MLP(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden=128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, output_dim)
#         )

#     def forward(self, x):
#         return self.net(x)

# class ReplayBuffer:
#     def __init__(self, capacity=20000):
#         self.buffer = deque(maxlen=capacity)

#     def push(self, s, a, r, s2, done):
#         self.buffer.append((s, a, r, s2, done))

#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         s, a, r, s2, d = zip(*batch)
#         return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
#                 np.array(s2), np.array(d, dtype=np.float32))

#     def __len__(self):
#         return len(self.buffer)

# class DQNAgent:
#     def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.99,
#                  epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
#                  device=None):
#         self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
#         self.model = MLP(obs_dim, action_dim).to(self.device)
#         self.target = MLP(obs_dim, action_dim).to(self.device)
#         self.target.load_state_dict(self.model.state_dict())
#         self.opt = optim.Adam(self.model.parameters(), lr=lr)
#         self.replay = ReplayBuffer()
#         self.gamma = gamma
#         self.epsilon = epsilon_start
#         self.epsilon_end = epsilon_end
#         self.epsilon_decay = epsilon_decay
#         self.action_dim = action_dim

#     def act(self, obs):
#         if random.random() < self.epsilon:
#             return random.randrange(self.action_dim)
#         obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             q = self.model(obs_t)
#         return int(q.argmax().cpu().numpy())

#     def push(self, s, a, r, s2, done):
#         self.replay.push(s, a, r, s2, done)

#     def update(self, batch_size=64):
#         if len(self.replay) < batch_size:
#             return
#         s, a, r, s2, d = self.replay.sample(batch_size)
#         s = torch.FloatTensor(s).to(self.device)
#         a = torch.LongTensor(a).to(self.device)
#         r = torch.FloatTensor(r).to(self.device)
#         s2 = torch.FloatTensor(s2).to(self.device)
#         d = torch.FloatTensor(d).to(self.device)

#         q = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
#         with torch.no_grad():
#             q2 = self.target(s2).max(1)[0]
#             target = r + (1 - d) * self.gamma * q2

#         loss = nn.MSELoss()(q, target)
#         self.opt.zero_grad()
#         loss.backward()
#         self.opt.step()

#     def sync_target(self):
#         self.target.load_state_dict(self.model.state_dict())

#     def decay_epsilon(self):
#         self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
# agent.py (Enhanced TrackMania-Lite DQN, Fixed MLP)



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# ===============================
# ✅ Neural Network (supports sensors)
# ===============================
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=256):  # deep enough for sensors
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ===============================
# ✅ Replay Memory
# ===============================
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(s2), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ===============================
# ✅ Double DQN Agent
# ===============================
class DQNAgent:
    def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 tau=0.01, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = MLP(obs_dim, action_dim).to(self.device)
        self.target = MLP(obs_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.model.state_dict())

        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.replay = ReplayBuffer()

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.action_dim = action_dim

    # ---------------------------------
    # ε-greedy action selection
    # ---------------------------------
    def act(self, obs):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.model(obs_t)
        return int(q.argmax().cpu().numpy())

    def push(self, s, a, r, s2, done):
        self.replay.push(s, a, r, s2, done)

    # ---------------------------------
    # ✅ Double DQN update
    # ---------------------------------
    def update(self, batch_size=64):
        if len(self.replay) < batch_size:
            return

        s, a, r, s2, d = self.replay.sample(batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        # Q(s, a)
        q = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN: use model to select, target to evaluate
        with torch.no_grad():
            next_actions = self.model(s2).argmax(1)
            next_q = self.target(s2).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = r + (1 - d) * self.gamma * next_q

        loss = nn.SmoothL1Loss()(q, target)  # Huber loss
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        # Soft target update
        for target_param, param in zip(self.target.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def sync_target(self):
        """Sync target network with the main network."""
        self.target.load_state_dict(self.model.state_dict())
        
    # ---------------------------------
    # Epsilon decay
    # ---------------------------------
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ---------------------------------
    # Save / Load models
    # ---------------------------------
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.model.state_dict())


