import gymnasium as gym
from stable_baselines3 import PPO  # A2C yerine PPO kullanacağız
from stable_baselines3.common.callbacks import EvalCallback
import pygame
import numpy as np

# CarRacing environmentini başlatıyoruz
env = gym.make("CarRacing-v2", render_mode='human')  # 'human' modda render alacağız

# PPO modelini oluşturuyoruz (hiperparametreler ayarlandı)
model = PPO("CnnPolicy", env, verbose=1, learning_rate=0.0001, ent_coef=0.01)

# Callback ile ara adımlarda log alıyoruz ve en iyi modeli kaydediyoruz
eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=10000)

# Modeli daha uzun süre eğitiyoruz (örnek olarak 100.000 adım)
model.learn(total_timesteps=100000, callback=eval_callback)

# PyGame ile ortamı görüntülemek için başlatıyoruz
pygame.init()

# Ortamın boyutlarına göre bir pencere oluşturuyoruz
screen_width, screen_height = 800, 600  # Bu değeri sabit tutalım
screen = pygame.display.set_mode((screen_width, screen_height))

# Çalışmayı durdurmak için bayrak
done = False

# Çevreyi sıfırla
obs, _ = env.reset()

# Simülasyon döngüsü
clock = pygame.time.Clock()  # FPS kontrolü için
fps = 60  # FPS değeri (ekran tazeleme hızı)
while not done:
    # Eğitimli modelden aksiyon al
    action, _states = model.predict(obs)
    
    # Ortamı bir adım ilerlet
    obs, rewards, terminated, truncated, info = env.step(action)
    
    # Ortam tamamlandığında çevreyi sıfırla
    if terminated or truncated:
        obs, _ = env.reset()
    
    # PyGame olaylarını kontrol et
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    
    # Görüntüle
    env.render()  # Pencereye görüntüyü doğrudan render ediyoruz
    
    # FPS kontrolü
    clock.tick(fps)

# Çevreyi kapat
env.close()
pygame.quit()
