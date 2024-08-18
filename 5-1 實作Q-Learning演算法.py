# 匯入套件
import gymnasium as gym  # OpenAI Gym 套件
import math
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

# 測試環境
env = gym.make('CartPole-v1', render_mode='rgb_array')  # 宣告環境
env.reset()  # 初始化
img = plt.imshow(env.render())  # 初始環境畫面
text = plt.text(0, 0, 'Timestamp   0', fontsize=20)  # 初始時間標示
plt.axis('off')  # 去除畫面軸線

for t in range(20):
    action = env.action_space.sample()  # 隨機選擇行動
    env.step(action)  # 行動
    img.set_data(env.render())  # 更新環境畫面
    text.set_text(f'Timestamp {t + 1:>3d}')  # 更新時間標示
    display.display(plt.gcf())  # IPython輸出畫面
    display.clear_output(wait=True)  # 清除畫面
env.close()  # 終止環境

# 單一行動測試
env = gym.make('CartPole-v1', render_mode='rgb_array')
observation, info = env.reset()  # 取得初始環境參數
episode_reward = 0  # 初始化獎勵
img = plt.imshow(env.render())
text = plt.text(0, 0, 'Timestamp   0', fontsize=20)
plt.axis('off')

for t in range(20):
    action = 1  # 固定action為1
    observation, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward  # 更新獎勵
    img.set_data(env.render())
    text.set_text(f'Timestamp {t + 1:>3d}')
    display.display(plt.gcf())
    display.clear_output(wait=True)
    if terminated or truncated:  # 環境終止條件
        break
env.close()  # 終止環境
print(f'Episode reward: {episode_reward}')  # 輸出獎勵

# 環境參數觀察
env = gym.make('CartPole-v1', render_mode='rgb_array')
observation, info = env.reset()
print(observation)


# 自訂決策函式
def decide_action(observation):
    x, v, theta, omega = observation
    return 0 if theta < 0 else 1


# 工人智慧測試
env = gym.make('CartPole-v1', render_mode='rgb_array')
episode_rewards = []
for i in range(200):
    observation, info = env.reset()
    episode_reward = 0
    while True:
        action = decide_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            break
    episode_rewards.append(episode_reward)

plt.clf()  # 清空畫框
plt.plot(episode_rewards)  # 輸出執行折線圖
plt.show()

# 定義狀態
env = gym.make('CartPole-v1', render_mode='rgb_array')
sections = (1, 1, 4, 3)  # (位置分段, 速度分段, 傾角分段, 角速度分段)
actions = env.action_space.n  # 行動總數(Cartpole中為2)

# 取得各參數範圍
state_bounds = list(
    zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-0.5, 0.5]  # 調整速度範圍
state_bounds[3] = [-math.radians(50), math.radians(50)]  #調整角速度範圍

# 環境參數轉Q表狀態函式
def get_state(observation, sections, state_bounds):
    state = [0] * len(observation)
    for i, s in enumerate(observation):
        mn, mx = state_bounds[i][0], state_bounds[i][1]
        if s > mx:  # 超過左界
            state[i] = sections[i] - 1
        elif s < mn:  # 超過右界
            state[i] = 0
        else:  # 尋找分段
            state[i] = int(((s-mn)/(mx-mn)) * sections[i])
    return tuple(state)

# 策略函式
def decide_action(state, q_table, action_space, epsilon):
    return np.argmax(
        q_table[state]
    ) if np.random.random_sample() > epsilon else action_space.sample()

# Q-Learning
def q_learning(epochs):
    env = gym.make('CartPole-v1')
    q_table = np.zeros(sections+(actions,))  # 建立空白Q表
    episode_rewards = []

    for i in range(epochs):

        epsilon = max(0.01, min(1.0, 1.0-math.log10((i+1)/25)))  # 更新隨機率
        lr = max(0.01, min(0.5, 1.0-math.log10((i+1)/25)))  # 更新學習率
        gamma = 0.999  # 折價率不變
        observation, info = env.reset()
        episode_reward = 0
        state = get_state(observation, sections, state_bounds)

        while True:
            action = decide_action(state, q_table, env.action_space, epsilon)
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = get_state(observation, sections, state_bounds)
            episode_reward += reward
            q_next_max = q_table[next_state].max()
            q_table[state][action] += lr * (reward
                                            + gamma*q_next_max
                                            - q_table[state][action])  # 更新Q表
            state = next_state
            if terminated or truncated:
                break
        episode_rewards.append(episode_reward)

    plt.plot(episode_rewards)
    plt.show()
    return q_table

q_star_table = q_learning(300) # 強化學習，得到Q*表

# 測試結果
env = gym.make('CartPole-v1', render_mode='rgb_array')
observation, info = env.reset()
episode_reward = 0
epsilon = 0
state = get_state(observation, sections, state_bounds)
img = plt.imshow(env.render())
text = plt.text(0, 0, 'Timestamp   0', fontsize=20)
plt.axis('off')

for t in range(500):
    action = decide_action(state, q_star_table, env.action_space, epsilon)
    observation, reward, terminated, truncated, info = env.step(action)
    state = get_state(observation, sections, state_bounds)
    episode_reward += reward
    img.set_data(env.render())
    text.set_text(f'Timestamp {t+1:>3d}')
    display.display(plt.gcf())
    display.clear_output(wait=True)
    if terminated or truncated:
        break
env.close()

print(f'Episode reward: {episode_reward}')
