import gymnasium as gym
import torch
from stable_baselines3 import SAC
import matplotlib.pyplot as plt
import time  # Eğitim süresi ölçümü için

# MountainCarContinuous ortamını yükle
env = gym.make('MountainCarContinuous-v0', render_mode="human")  # render_mode="human" ile görselleştirme aktif

# SAC modelini tanımla
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_mountaincar_tensorboard/")

# Eğitim adedi ve episod sonuçlarını toplamak için değişkenler
total_timesteps = 50000
log_interval = 750
episode_rewards = []

# Her bir episodun maksimum adım sayısı (örneğin 1000 adım)
max_steps_per_episode = 750

# Eğitim süresi ölçümü için başlangıç zamanı
start_time = time.time()

# Modeli eğit
def train_and_log(model, timesteps):
    obs, _ = env.reset()
    episode_reward = 0
    steps_in_current_episode = 0  # Bu değişken her episodun adım sayısını takip eder
    count_done = 0
    i = 0
    
    while i < timesteps:
        i += 1
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, _, _ = env.step(action)
        episode_reward += reward
        steps_in_current_episode += 1
        
        # Eğer adımlar maksimuma ulaşırsa ya da "done" olursa episod sıfırlanır
        if done or steps_in_current_episode >= max_steps_per_episode:
            if done:
                count_done += 1
                print("Başarıyla tamamlandı!")
            episode_rewards.append(episode_reward)
            episode_reward = 0
            steps_in_current_episode = 0  # Adım sayısını sıfırla
            obs, _ = env.reset()
            
        # Her 50 adımda bir ortalama reward yazdır
        if (i + 1) % log_interval == 0:
            recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
            if len(recent_rewards) > 0:
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                print(f"Adım: {i + 1}, Ortalama Reward: {round(avg_reward, 2)}")
    
    print("Eğitim tamamlandı, başarıyla tamamlanan episodlar: ", count_done)
    return {"episode_rewards": episode_rewards, "count_done": count_done}

# Eğitim ve episodları logla
results = train_and_log(model, total_timesteps)

# Eğitim süresi ölçümü için bitiş zamanı
end_time = time.time()
training_time = end_time - start_time

# Sonuçları al
episode_rewards = results["episode_rewards"]
count_done = results["count_done"]

# Eğitim sonrası sonuçları görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('SAC Training on MountainCarContinuous')

# Eğitim süresi, toplam ödül ve diğer bilgileri grafiğe ekle
num_episodes = len(episode_rewards)
total_reward = sum(episode_rewards)
plt.text(0.95, 0.05, f'Episodes: {num_episodes}', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)
plt.text(0.95, 0.10, f'Total Reward: {round(total_reward, 2)}', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)
plt.text(0.95, 0.20, f'Avg Reward: {round(total_reward/len(episode_rewards), 2)}', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)
plt.text(0.95, 0.25, f'Count Done: {round(count_done, 2)}', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)
plt.text(0.95, 0.15, f'Training Time: {round(training_time, 2)} sec', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)

plt.show()

# Eğitim süresi, toplam ödül ve episod bilgilerini yazdır
print(f"Eğitim süresi: {round(training_time, 2)} saniye")
print(f"Toplam ödül: {round(total_reward, 2)}")
print(f"Tamamlanan episode sayısı: {num_episodes}")