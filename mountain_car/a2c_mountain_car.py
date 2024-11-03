import gymnasium as gym
import time
import torch
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

# Ödülleri toplamak ve adım sayacını kullanmak için callback tanımlama
class RewardCallback(BaseCallback):
    def __init__(self, log_interval=300, reset_interval=500):
        super(RewardCallback, self).__init__()
        self.episode_rewards = []
        self.log_interval = log_interval
        self.reset_interval = reset_interval
        self.step_count = 0  # Adım sayacı
        self.current_episode_reward = 0  # Mevcut episod ödülü
        self.done_count=0

    def _on_step(self) -> bool:
        # Env'den ödülleri toplamak
        self.step_count += 1
        reward = self.locals['rewards'][0]
        self.current_episode_reward += reward

        if self.current_episode_reward<-60:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0  
            self.locals['env'].reset()
            print("750 adım tamamlandı, ortam sıfırlanıyor.")
            
        if self.step_count % self.log_interval == 0:
            print(self.episode_rewards)
            print(f"Adım: {self.step_count}, Son 10 Episode Toplam Reward: {sum(self.episode_rewards[-10:])}, Done: {self.done_count}")

        # Eğer bir episod tamamlandıysa, ödülü kaydet
        if self.locals['dones'][0]:
            self.done_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0  # Mevcut episod ödülünü sıfırla
        
        # Her 750 adımda bir ortamı sıfırla
        if self.step_count % self.reset_interval == 0:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0  
            print("750 adım tamamlandı, ortam sıfırlanıyor.")
            self.locals['env'].reset()
        
        return True

# CarRacing ortamını yükle
env = gym.make('MountainCarContinuous-v0', render_mode="human")  # Görselleştirme için render_mode="human"

# A2C modelini tanımla
model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./a2c_mountaincar_tensorboard/")

# Eğitim parametreleri
total_timesteps = 50000
log_interval = 50

# Callback örneğini oluştur
reward_callback = RewardCallback(log_interval=250, reset_interval=750)

# Eğitim süresi ölçümü için başlangıç zamanı
start_time = time.time()

# Modeli eğit
model.learn(total_timesteps=total_timesteps, log_interval=log_interval, callback=reward_callback)

# Eğitim süresi ölçümü için bitiş zamanı
end_time = time.time()
training_time = end_time - start_time

# Eğitim sonrası sonuçları görselleştirme
episode_rewards = reward_callback.episode_rewards
total_reward = sum(episode_rewards)

plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, label="Episode Reward")
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('A2C Training on Mountain Car')

# Eğitim süresi, toplam ödül ve diğer bilgileri grafiğe ekle
plt.text(0.95, 0.05, f'Total Reward: {round(total_reward, 2)}', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)
plt.text(0.95, 0.10, f'Training Time: {round(training_time, 2)} sec', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)
plt.text(0.95, 0.15, f'Average Reward: {round(sum(episode_rewards)/len(episode_rewards), 2)}', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)

plt.legend()
plt.show()

# Eğitim süresi bilgilerini yazdır
print(f"Eğitim süresi: {round(training_time, 2)} saniye")
print(f"Toplam ödül: {round(total_reward, 2)}")

# Çevreyi kapat
env.close()
