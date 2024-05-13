# q-learning

import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions, gamma=0.9, alpha=0.1, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = np.random.uniform(-1, 1, size=(n_states, n_actions))
    
    def choose_action(self, state): # epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state):
        target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    def solve(self, env, n_episodes=1000, max_steps=1000):
        for episode in range(n_episodes):
            t = 0
            state, _ = env.reset()
            done = False
            while t < max_steps and not done:
                action = self.choose_action(state)
                next_state, reward, done, *_ = env.step(action)
                # print(state, action, reward, next_state)
                self.update(state, action, reward, next_state)
                state = next_state

    def get_policy(self):
        return np.argmax(self.Q, axis=1)
    
    def play(self, env):
        state, _ = env.reset()
        done = False
        while not done:
            action = self.get_policy()[state]
            state, _, done, *_ = env.step(action)
            env.render()


# Test
if __name__ == "__main__":

    import gym
    env = gym.make("Taxi-v3"
                #    , render_mode="human"
                   )

    env.action_space.seed(42)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    qlearning = QLearning(n_states, n_actions, gamma=0.9, alpha=0.1, epsilon=0.1)
    qlearning.solve(env, n_episodes=1000, max_steps=1000)
    print(qlearning.Q)
    print(qlearning.get_policy())

    env.close()

    env = gym.make("Taxi-v3", render_mode="human")
    qlearning.play(env)

    env.close()