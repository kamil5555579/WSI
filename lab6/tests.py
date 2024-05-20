from solver import QLearning
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3")
n_states = env.observation_space.n
n_actions = env.action_space.n

alphas = [0.1, 0.5, 1]
strategies = ["epsilon-greedy", "boltzmann"]
epsilons = [0.1, 0.2, 0.5]
temperatures = [0.1, 1, 5]

# for epsilon in epsilons:
#     qlearning = QLearning(n_states, n_actions, gamma=0.9, alpha=1, epsilon=epsilon)
#     qlearning.solve(env, n_episodes=2000, max_steps=500)
#     plt.plot(qlearning.episode_rewards, label=f"epsilon={epsilon}")
#     print(qlearning.evaluate(env, n_episodes=500, max_steps=500))

# plt.legend()
# plt.xlabel("Episodes")
# plt.ylabel("Total rewards")
# plt.show()

# for temperature in temperatures:
#     qlearning = QLearning(n_states, n_actions, gamma=0.9, alpha=1, temperature=temperature, strategy="boltzmann")
#     qlearning.solve(env, n_episodes=2000, max_steps=500)
#     plt.plot(qlearning.episode_rewards, label=f"temperature={temperature}")
#     print(qlearning.evaluate(env, n_episodes=500, max_steps=500))

# plt.legend()
# plt.xlabel("Episodes")
# plt.ylabel("Total rewards")
# plt.show()

for strategy in strategies:
    qlearning = QLearning(n_states, n_actions, gamma=0.9, alpha=1, epsilon=0.1, strategy=strategy)
    qlearning.solve(env, n_episodes=2000, max_steps=500)
    plt.plot(qlearning.episode_rewards, label=strategy)
    print(qlearning.evaluate(env, n_episodes=500, max_steps=500))

plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Total rewards")
plt.show()

# for alpha in alphas:
#     qlearning = QLearning(n_states, n_actions, gamma=0.9, alpha=alpha, epsilon=0.1)
#     qlearning.solve(env, n_episodes=2000, max_steps=500)
#     plt.plot(qlearning.episode_rewards, label=f"alpha={alpha}")
#     print(qlearning.evaluate(env, n_episodes=500, max_steps=500))

# plt.legend()
# plt.xlabel("Episodes")
# plt.ylabel("Total rewards")
# plt.show()