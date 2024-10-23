import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
import matplotlib.pyplot as plt

# Blackjack ortamını yükle
env = gym.make('Blackjack-v1', render_mode="human")

# Gözlem uzayını uygun bir formata dönüştürmek için wrapper fonksiyonu
class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Gözlem uzayını düzleştiriyoruz
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

    def observation(self, obs):
        # Gözlemleri (oyuncunun eli, krupiyenin kartı, as sayısı) düzleştiriyoruz
        return np.array(obs, dtype=np.float32)

# Ortamı sarıp gözlemleri düzleştirme
env = FlattenObservation(env)

# A2C modelini tanımla (MlpPolicy kullanılacak çünkü ortam küçük ve tablo bazlı)
model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./a2c_blackjack_tensorboard/")

# Eğitim adedi ve episod sonuçlarını toplamak için değişkenler
total_timesteps = 1000
log_interval = 1000
episode_rewards = []

# Dealer ve oyuncu kazanma sayıları
dealer_wins = 0
player_wins = 0

# Kazanma sayılarını saklamak için listeler
dealer_wins_list = []
player_wins_list = []

# Her bir episodun maksimum adım sayısı
max_steps_per_episode = 100

# Modeli eğit
def train_and_log(model, timesteps):
    obs, _ = env.reset()
    episode_reward = 0
    episode_done_count = 0
    step_count = 0
    
    dealer_wins = 0
    player_wins = 0

    i = 0

    while i < total_timesteps:  # 1000 başarılı episod tamamlanana kadar eğitime devam et
        step_count += 1
        i += 1
        action, _ = model.predict(obs, deterministic=False)
        # Ortama aksiyonu gönder
        obs, reward, done, _, info = env.step(action)
        episode_reward += reward

        # Eğer adımlar maksimuma ulaşırsa ya da "done" olursa episod sıfırlanır
        if done or step_count >= max_steps_per_episode:
            episode_rewards.append(episode_reward)
            if done:
                episode_done_count += 1
                # Dealer kazandığında kontrol et
                if reward < 0:  # Eğer ödül negatifse, dealer kazanmıştır
                    dealer_wins += 1
                else:
                    player_wins += 1

            # Kazanma sayılarını listelere ekle
            dealer_wins_list.append(dealer_wins)
            player_wins_list.append(player_wins)

            episode_reward = 0  # Episod ödülünü sıfırla
            step_count = 0  # Adım sayısını sıfırla
            obs, _ = env.reset()

        print(i, "dealer:", dealer_wins, "player:", player_wins)

        # Her log_interval'da bir ortalama reward yazdır
        if episode_done_count % log_interval == 0:
            recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
            if len(recent_rewards) > 0:
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                print(f"Episod: {episode_done_count}, Ortalama Reward: {round(avg_reward, 2)}")

    return episode_rewards

# Modeli eğit ve sonuçları logla
episode_rewards = train_and_log(model, total_timesteps)

# Eğitim sonrası başarı oranını hesapla
total_episodes = len(episode_rewards)
avg_reward = sum(episode_rewards) / total_episodes if total_episodes > 0 else 0
print(f"Ortalama Reward: {avg_reward:.2f}")

# Eğitim sonrası sonuçları görselleştirme
# İlk grafikte episode rewardları
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, label='Episode Rewards', color='blue')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Rewards Over Time')
plt.legend()
plt.show()

# İkinci grafikte dealer ve player kazanma sayılarını
plt.figure(figsize=(10, 6))
plt.plot(dealer_wins_list, label='Dealer Wins', color='red', linestyle='--')
plt.plot(player_wins_list, label='Player Wins', color='blue', linestyle='--')
plt.xlabel('Episode')
plt.ylabel('Wins')
plt.title('Dealer and Player Wins Over Time')
plt.legend()
plt.show()

# Eğitim sonrası modelin kaydedilmesi
model.save("a2c_blackjack_v1")

# Render'ı kapatmak için
env.close()
