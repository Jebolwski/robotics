import gymnasium as gym
import time
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

# Define callback to collect rewards and log FPS every 5 steps
class RewardCallback(BaseCallback):
    def __init__(self, log_interval=300, reset_interval=500, fps_log_interval=20000):
        super(RewardCallback, self).__init__()
        self.episode_rewards = []
        self.log_interval = log_interval
        self.reset_interval = reset_interval
        self.fps_log_interval = fps_log_interval
        self.step_count = 0
        self.current_episode_reward = 0
        self.last_time = time.time()  # Start time for FPS calculation

    def _on_step(self) -> bool:
        # Increment step count and accumulate reward
        self.step_count += 1
        reward = self.locals['rewards'][0]
        self.current_episode_reward += reward

        # Reset environment if reward is below threshold
        if self.current_episode_reward < -60:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0  
            self.locals['env'].reset()
        
        # Log total reward every log_interval steps
        if self.step_count % self.log_interval == 0:
            print(self.episode_rewards)
            print(f"Adım: {self.step_count}, Son 10 Episode Toplam Reward: {sum(self.episode_rewards[-10:])}")

        # Track episode completion and reset episode reward
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
        
        # Reset environment every reset_interval steps
        if self.step_count % self.reset_interval == 0:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0  
            print("500 adım tamamlandı, ortam sıfırlanıyor.")
            self.locals['env'].reset()
        
        return True

# Load CarRacing environment
env = gym.make('CarRacing-v2', render_mode="human")  # render_mode="human" for visualization

# Define SAC model
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_carracing_tensorboard/",
            buffer_size=5000, train_freq=(1, "episode"), learning_starts=500)


# Training parameters
total_timesteps = 15000
log_interval = 50

# Instantiate callback with logging intervals
reward_callback = RewardCallback(log_interval=250, reset_interval=750, fps_log_interval=5)

# Record training start time
start_time = time.time()

# Train model
model.learn(total_timesteps=total_timesteps, log_interval=log_interval, callback=reward_callback)

# Record training end time and calculate total duration
end_time = time.time()
training_time = end_time - start_time

# Plot training results
episode_rewards = reward_callback.episode_rewards
total_reward = sum(episode_rewards)

plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, label="Episode Reward")
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('SAC Training on CarRacing')

# Display training stats on the plot
plt.text(0.95, 0.05, f'Total Reward: {round(total_reward, 2)}', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)
plt.text(0.95, 0.10, f'Training Time: {round(training_time, 2)} sec', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)
plt.text(0.95, 0.15, f'Average Reward: {round(sum(episode_rewards)/len(episode_rewards), 2)}', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)

plt.legend()
plt.show()

# Display training time and total reward
print(f"Eğitim süresi: {round(training_time, 2)} saniye")
print(f"Toplam ödül: {round(total_reward, 2)}")

# Close the environment
env.close()
