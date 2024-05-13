import gym
env = gym.make("Taxi-v3", render_mode="human")

env.action_space.seed(42)

observation, info = env.reset(seed=42)
print(observation, info)
print(env.truncated, env.terminated)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(action)
    print(observation, reward, terminated, truncated, info)

    if terminated or truncated:
        observation, info = env.reset()

env.close()