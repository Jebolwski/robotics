import gymnasium as gym
import torch
from stable_baselines3 import SAC
import matplotlib.pyplot as plt

# MountainCarContinuous ortamını yükle
env = gym.make('MountainCarContinuous-v0', render_mode="human")  # render_mode="human" ile görselleştirme aktif

# PPO modelini tanımla
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_mountaincar_tensorboard/")

# Eğitim adedi ve episod sonuçlarını toplamak için değişkenler
total_timesteps = 20000
log_interval = 50
episode_rewards = []

# Her bir episodun maksimum adım sayısı (örneğin 1000 adım)
max_steps_per_episode = 2000

# Modeli eğit
def train_and_log(model, timesteps):
    obs, _ = env.reset()
    episode_reward = 0
    steps_in_current_episode = 0  # Bu değişken her episodun adım sayısını takip eder
    count_done=0
    i=0
    while count_done<4:
        i+=1
        # env.render()  # Bu satır ile her adımı görselleştiririz
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, _, _ = env.step(action)
        episode_reward += reward
        steps_in_current_episode += 1
        # Eğer adımlar maksimuma ulaşırsa ya da "done" olursa episod sıfırlanır
        if done or steps_in_current_episode >= max_steps_per_episode:
            if (done):
                count_done+=1
                print("VALLA OLDU")
            episode_rewards.append(episode_reward)
            episode_reward = 0
            steps_in_current_episode = 0  # Adım sayısını sıfırla
            obs, _ = env.reset()
            
        # Her 50 adımda bir ortalama reward yazdır
        if (i + 1) % log_interval == 0:
            # Eğer yeterince episode yoksa, tüm episode'ları al
            recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
            if len(recent_rewards) > 0:
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                print(f"Adım: {i + 1}, Ortalama Reward: {round(avg_reward,2)}")
    print("oldu ",count_done)
    return {episode_rewards,count_done}

# Eğitim ve episodları görselleştirme
episode_rewards,count_done = train_and_log(model, total_timesteps)

# Eğitim sonrası sonuçları görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('PPO Training on MountainCarContinuous')

# Episode numarasını sağ alt köşeye ekle
num_episodes = len(episode_rewards)
plt.text(0.95, 0.05, f'Episodes: {num_episodes}', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)
plt.text(0.95, 0.10, f'Olma Sayısı: {count_done}', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)

plt.show()

# Eğitim sonrası modelin kaydedilmesi
model.save("ppo_mountaincar_continuous")

# Render'ı kapatmak için
env.close()
