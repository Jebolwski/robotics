import gymnasium as gym
import torch
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import time  # Eğitim süresi ölçümü için

# CarRacing ortamını yükle
env = gym.make('CarRacing-v2', render_mode="human")

# PPO modelini tanımla
model = PPO("CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./a2c_carracing_tensorboard/",
    learning_rate=0.0003,
    gamma=0.98,
    n_steps=5,
    gae_lambda=0.9)

# Eğitim adedi ve episod sonuçlarını toplamak için değişkenler
total_timesteps = 50000
log_interval = 100
episode_rewards = []
successful_episodes = 0  # Başarılı episodları saymak için
success_threshold = 900  # Başarı için gereken ödül eşiği

# Her bir episodun maksimum adım sayısı
max_steps_per_episode = 500

# Eğitim süresi ölçümü için başlangıç zamanı
start_time = time.time()

# FPS ayarı
target_fps = 90
frame_time = 1.0 / target_fps  # Her frame için geçen süre (60 FPS için ~0.0167 saniye)

# Modeli eğit
def train_and_log(model, timesteps):
    obs, _ = env.reset()
    episode_reward = 0
    steps_in_current_episode = 0
    episode_done_count = 0
    step_count = 0
    global successful_episodes  # Global başarı sayacı

    while step_count < total_timesteps:
        step_count += 1
        action, _ = model.predict(obs, deterministic=False)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        # Başlama zamanı
        start_frame_time = time.time()

        obs, reward, done, _, _ = env.step(action)
        episode_reward += reward
        steps_in_current_episode += 1

        # Eğer adımlar maksimuma ulaşırsa ya da "done" olursa episod sıfırlanır
        if done or steps_in_current_episode >= max_steps_per_episode:
            episode_rewards.append(episode_reward)
            episode_done_count += 1 if done else 0

            # Başarılı episodları kontrol et
            if episode_reward >= success_threshold:
                successful_episodes += 1  # Başarılı episodları say

            episode_reward = 0
            steps_in_current_episode = 0  # Adım sayısını sıfırla
            obs, _ = env.reset()

        # Her log_interval'da bir ortalama reward yazdır
        if step_count % log_interval == 0:
            recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
            if len(recent_rewards) > 0:
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                print(f"Adım: {step_count}, Ortalama Reward: {round(avg_reward, 2)}")

        # Render'ı göster
        env.render()

        # Frame süresi kadar bekle (60 FPS ayarı)
        elapsed_time = time.time() - start_frame_time
        if elapsed_time < frame_time:
            time.sleep(frame_time - elapsed_time)

    return episode_rewards

# Modeli eğit ve sonuçları logla
episode_rewards = train_and_log(model, total_timesteps)

# Eğitim süresi ölçümü için bitiş zamanı
end_time = time.time()
training_time = end_time - start_time

# Toplam ödülü hesapla
total_reward = sum(episode_rewards)
print(f"Toplam Ödül: {total_reward}")

# Eğitim sonrası sonuçları görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('PPO Training on CarRacing-v2')

# Episode numarasını ve toplam ödülü sağ alt köşeye ekle
num_episodes = len(episode_rewards)
plt.text(0.95, 0.05, f'Episodes: {num_episodes}', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)
plt.text(0.95, 0.10, f'Total Reward: {total_reward:.2f}', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)
plt.text(0.95, 0.15, f'Training Time: {round(training_time, 2)} sec', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)

plt.show()

# Eğitim sonrası modelin kaydedilmesi
model.save("ppo_carracing_v2")

# Render'ı kapatmak için
env.close()

# Eğitim süresi ve başarı sayısını yazdır
print(f"Eğitim süresi: {round(training_time, 2)} saniye")
print(f"Başarıyla tamamlanan episod sayısı: {successful_episodes}")
print(f"Tamamlanan episode sayısı: {num_episodes}")
