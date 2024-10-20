import gymnasium as gym
import torch
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt

# CarRacing ortamını yükle
env = gym.make('CarRacing-v2', render_mode="human")  # render_mode="human" ile görselleştirme aktif

# PPO modelini tanımla
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_carracing_tensorboard/")

# Eğitim adedi ve episod sonuçlarını toplamak için değişkenler
total_timesteps = 50000
log_interval = 100
episode_rewards = []

# Her bir episodun maksimum adım sayısı (örneğin 1000 adım)
max_steps_per_episode = 1000

# Modeli eğit
def train_and_log(model, timesteps):
    obs, _ = env.reset()
    episode_reward = 0
    steps_in_current_episode = 0  # Bu değişken her episodun adım sayısını takip eder
    episode_done_count = 0
    step_count = 0

    while episode_done_count < 4:  # 4 episod tamamlanana kadar eğitime devam et
        step_count += 1
        action, _ = model.predict(obs, deterministic=False)

        # Aksiyonları ortamın aksiyon uzayına göre sınırla (normalleştir)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        # Ortama aksiyonu ver
        obs, reward, done, _, _ = env.step(action)
        episode_reward += reward
        steps_in_current_episode += 1

        # Eğer adımlar maksimuma ulaşırsa ya da "done" olursa episod sıfırlanır
        if done or steps_in_current_episode >= max_steps_per_episode:
            episode_rewards.append(episode_reward)
            episode_done_count += 1 if done else 0
            episode_reward = 0
            steps_in_current_episode = 0  # Adım sayısını sıfırla
            obs, _ = env.reset()

        # Her log_interval'da bir ortalama reward yazdır
        if step_count % log_interval == 0:
            # Eğer yeterince episode yoksa, tüm episode'ları al
            recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
            if len(recent_rewards) > 0:
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                print(f"Adım: {step_count}, Ortalama Reward: {round(avg_reward, 2)}")

    return episode_rewards

# Modeli eğit ve sonuçları logla
episode_rewards = train_and_log(model, total_timesteps)

# Eğitim sonrası sonuçları görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('PPO Training on CarRacing-v2')

# Episode numarasını sağ alt köşeye ekle
num_episodes = len(episode_rewards)
plt.text(0.95, 0.05, f'Episodes: {num_episodes}', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)

plt.show()

# Eğitim sonrası modelin kaydedilmesi
model.save("ppo_carracing_v2")

# Render'ı kapatmak için
env.close()
