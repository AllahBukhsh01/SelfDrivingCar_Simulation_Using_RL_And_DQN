# import torch
# import matplotlib.pyplot as plt
# from car_env import CarEnv
# from agent import DQNAgent
# import numpy as np

# def train_dqn(episodes=200):
#     env = CarEnv(render_mode=True)
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n
#     agent = DQNAgent(state_dim, action_dim)

#     rewards = []

#     for ep in range(episodes):
#         state = env.reset()
#         total_reward = 0
#         done = False

#         while not done:
#             action = agent.act(state)
#             next_state, reward, done, _ = env.step(action)
#             agent.memory.push(state, action, reward, next_state, done)
#             agent.update(batch_size=64)
#             state = next_state
#             total_reward += reward

#         agent.update_target()
#         agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
#         rewards.append(total_reward)

#         print(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

#     env.close()
#     return rewards


# def plot_rewards(rewards):
#     plt.plot(rewards)
#     plt.xlabel("Episode")
#     plt.ylabel("Total Reward")
#     plt.title("Training Progress - DQN Self-Driving Car")
#     plt.show()


# if __name__ == "__main__":
#     rewards = train_dqn(episodes=200)
#     plot_rewards(rewards)


#=================================================================================================

#                                     REAL TRACK VERSION

#=================================================================================================


# # main.py
# import time
# import numpy as np
# import matplotlib.pyplot as plt

# from car_env import CarEnv
# from agent import DQNAgent

# def train(episodes=300, max_steps=1200):
#     env = CarEnv(render_mode=True)
#     obs_dim = env.observation_dim
#     action_dim = env.action_space

#     agent = DQNAgent(obs_dim, action_dim, lr=1e-3)
#     rewards_hist = []

#     for ep in range(1, episodes+1):
#         obs = env.reset()
#         total = 0.0
#         done = False
#         steps = 0

#         while not done and steps < max_steps:
#             action = agent.act(obs)
#             next_obs, reward, done, info = env.step(action)
#             agent.push(obs, action, reward, next_obs, done)
#             agent.update(batch_size=64)

#             obs = next_obs
#             total += reward
#             steps += 1

#             # render each step (keeps window interactive)
#             env.render()

#         agent.sync_target()
#         agent.decay_epsilon()
#         rewards_hist.append(total)
#         print(f"Episode {ep}/{episodes} | Reward: {total:.2f} | Epsilon: {agent.epsilon:.3f}")

#     env.close()
#     return rewards_hist, agent

# def plot_rewards(rewards):
#     plt.figure(figsize=(10,5))
#     plt.plot(rewards)
#     plt.xlabel("Episode")
#     plt.ylabel("Total Reward")
#     plt.title("Training - TrackMania-Lite (DQN)")
#     plt.grid(True)
#     plt.show()

# if __name__ == "__main__":
#     r, agent = train(episodes=200)
#     plot_rewards(r)
#     # Optionally save model
#     import torch
#     torch.save(agent.model.state_dict(), "dqn_trackmania.pth")
#     print("Model saved to dqn_trackmania.pth")


#==================================================================================================


#==================================================================================================

# # main.py
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# import torch

# from car_env import CarEnv
# from agent import DQNAgent

# def train(episodes=250, max_steps=1500):
#     env = CarEnv(render_mode=True)
#     obs_dim = env.observation_dim
#     action_dim = env.action_space

#     agent = DQNAgent(obs_dim, action_dim, lr=1e-3)
#     rewards_hist = []

#     for ep in range(1, episodes+1):
#         obs = env.reset()
#         total = 0.0
#         done = False
#         steps = 0

#         while not done and steps < max_steps:
#             action = agent.act(obs)
#             next_obs, reward, done, info = env.step(action)
#             agent.push(obs, action, reward, next_obs, done)
#             agent.update(batch_size=64)

#             obs = next_obs
#             total += reward
#             steps += 1

#             # render to keep window interactive and visual
#             env.render()

#         # Post-episode updates
#         agent.sync_target()
#         agent.decay_epsilon()
#         rewards_hist.append(total)
#         print(f"Episode {ep}/{episodes} | Reward: {total:.2f} | Epsilon: {agent.epsilon:.3f} | Passed CPs: {len(env.passed_checkpoints)}")

#         # quick checkpointing save every 50 episodes
#         if ep % 50 == 0:
#             torch.save(agent.model.state_dict(), f"dqn_trackmania_ep{ep}.pth")
#             print(f"[Main] Saved model at episode {ep}")

#     env.close()
#     return rewards_hist, agent

# def plot_rewards(rewards):
#     plt.figure(figsize=(10,5))
#     plt.plot(rewards, color="tab:orange")
#     plt.xlabel("Episode")
#     plt.ylabel("Total Reward")
#     plt.title("Training - TrackMania-Lite (DQN)")
#     plt.grid(True)
#     plt.show()

# if __name__ == "__main__":
#     rewards, agent = train(episodes=200)
#     plot_rewards(rewards)
#     # final model save
#     torch.save(agent.model.state_dict(), "dqn_trackmania_final.pth")
#     print("Saved final model: dqn_trackmania_final.pth")



# main.py
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

from car_env import CarEnv
from agent import DQNAgent

def draw_sensors(env, color=(0,255,0)):
    """Draw sensor rays (visualize what agent sees)."""
    if not hasattr(env, 'car_pos'): 
        return
    sensors = env._sense_environment()
    sensor_angles = [0, -30, 30, -60, 60]
    max_range = 150
    for rel_ang, strength in zip(sensor_angles, sensors):
        ang = np.radians(env.car_angle + rel_ang)
        dist = strength * max_range
        end_x = int(env.car_pos[0] + dist * np.cos(ang))
        end_y = int(env.car_pos[1] - dist * np.sin(ang))
        pygame = __import__('pygame')
        pygame.draw.line(env.screen, color, (int(env.car_pos[0]), int(env.car_pos[1])), (end_x, end_y), 2)
        pygame.draw.circle(env.screen, (255,255,0), (end_x, end_y), 3)

def train(episodes=250, max_steps=1500, render=True):
    env = CarEnv(render_mode=render)
    obs_dim = env.observation_dim
    action_dim = env.action_space

    agent = DQNAgent(obs_dim, action_dim, lr=1e-3)
    rewards_hist = []
    smoothed_rewards = []

    for ep in range(1, episodes+1):
        obs = env.reset()
        total = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            agent.push(obs, action, reward, next_obs, done)
            agent.update(batch_size=64)

            obs = next_obs
            total += reward
            steps += 1

            if render:
                env.render()
                draw_sensors(env)  # visualize sensors

        # Sync & decay after each episode
        agent.sync_target()
        agent.decay_epsilon()
        rewards_hist.append(total)
        smoothed = np.mean(rewards_hist[-10:])  # moving average
        smoothed_rewards.append(smoothed)

        print(f"Episode {ep}/{episodes} | Reward: {total:.2f} | Avg(10): {smoothed:.2f} | Epsilon: {agent.epsilon:.3f} | CPs: {len(env.passed_checkpoints)}")

        # checkpoint save
        if ep % 50 == 0:
            torch.save(agent.model.state_dict(), f"dqn_trackmania_New_ep{ep}.pth")
            print(f"[Main] ✅ Saved model at episode {ep}")

    env.close()
    return rewards_hist, smoothed_rewards, agent


def plot_rewards(raw, smooth):
    plt.figure(figsize=(10,5))
    plt.plot(raw, color="orange", alpha=0.4, label="Raw")
    plt.plot(smooth, color="red", linewidth=2, label="Smoothed (avg 10)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("TrackMania-Lite (DQN Sensor-Based Training)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    rewards, smooth, agent = train(episodes=200, render=True)
    plot_rewards(rewards, smooth)
    torch.save(agent.model.state_dict(), "dqn_trackmania_final.pth")
    print("✅ Saved final model: dqn_trackmania_final.pth")
