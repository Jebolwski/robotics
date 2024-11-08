import gymnasium as gym
import torch
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import time  # For measuring training duration

# Load the CarRacing environment
env = gym.make('MountainCarContinuous-v0', render_mode="human")  # Use render_mode="human" for visualization

# Define the PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_mountaincar_tensorboard/")

# Variables to keep track of training steps and episode results
total_timesteps = 50000
log_interval = 750
episode_rewards = []

# Maximum number of steps per episode (e.g., 1000 steps)
max_steps_per_episode = 750

# Start time measurement for training duration
start_time = time.time()

# Train and log function
def train_and_log(model, timesteps):
    obs, _ = env.reset()
    episode_reward = 0
    steps_in_current_episode = 0  # This variable tracks the step count for each episode
    count_done = 0
    i = 0
    
    while i < timesteps:
        i += 1
        action = model.predict(obs, deterministic=False)[0]
        obs, reward, done, _, _ = env.step(action)
        episode_reward += reward
        steps_in_current_episode += 1
        
        # Reset episode if maximum steps reached or "done" condition met
        if done or steps_in_current_episode >= max_steps_per_episode:
            if done:
                count_done += 1
                print("Successfully completed an episode!")
            episode_rewards.append(episode_reward)
            episode_reward = 0
            steps_in_current_episode = 0  # Reset step count
            obs, _ = env.reset()
            
        # Print average reward every log_interval steps
        if (i + 1) % log_interval == 0:
            recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
            if len(recent_rewards) > 0:
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                print(f"Step: {i + 1}, Average Reward: {round(avg_reward, 2)}")
    
    print("Training completed, successfully completed episodes: ", count_done)
    return {"episode_rewards": episode_rewards, "count_done": count_done}

# Train and log episodes
results = train_and_log(model, total_timesteps)

# Measure training duration
end_time = time.time()
training_time = end_time - start_time

# Retrieve results
episode_rewards = results["episode_rewards"]
count_done = results["count_done"]

# Visualization of training results
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('PPO Training on CarRacing')

# Add training duration, total reward, and other information to the plot
num_episodes = len(episode_rewards)
total_reward = sum(episode_rewards)
plt.text(0.95, 0.05, f'Episodes: {num_episodes}', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)
plt.text(0.95, 0.10, f'Total Reward: {round(total_reward, 2)}', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)
plt.text(0.95, 0.20, f'Avg Reward: {round(total_reward/len(episode_rewards), 2)}', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)
plt.text(0.95, 0.25, f'Count Done: {round(count_done, 2)}', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)
plt.text(0.95, 0.15, f'Training Time: {round(training_time, 2)} sec', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)

plt.show()

# Print training duration, total reward, and episode information
print(f"Training duration: {round(training_time, 2)} seconds")
print(f"Total reward: {round(total_reward, 2)}")
print(f"Number of completed episodes: {num_episodes}")
