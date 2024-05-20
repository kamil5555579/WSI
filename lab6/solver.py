import numpy as np


class QLearning:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.9,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        temperature: float = 1.0,
        strategy: str = "epsilon-greedy",
    ):
        """
        n_states: number of states
        n_actions: number of actions
        gamma: discount factor
        alpha: learning rate
        epsilon: exploration rate
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.temperature = temperature
        self.strategy = strategy
        self.Q = np.random.uniform(-1, 1, size=(n_states, n_actions))

    def choose_action(self, state: int, strategy: str = "epsilon-greedy") -> int:
        """
        Choose an action based on current state and strategy
        """
        if strategy == "epsilon-greedy":
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.n_actions)
            return np.argmax(self.Q[state])
        elif strategy == "boltzmann":
            q_values = self.Q[state]
            probabilities = np.exp(q_values / self.temperature) / np.sum(
                np.exp(q_values / self.temperature)
            )
            return np.random.choice(self.n_actions, p=probabilities)

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Update Q-values
        """
        delta = reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action]
        self.Q[state, action] += self.alpha * delta

    def solve(
        self, env: "gym.Env", n_episodes: int = 1000, max_steps: int = 1000
    ) -> None:
        """
        Solve the given environment
        """
        self.episode_rewards = []
        for episode in range(n_episodes):
            t = 0
            state, _ = env.reset()
            done = False
            total_reward = 0
            while t < max_steps and not done:
                action = self.choose_action(state, strategy=self.strategy)
                next_state, reward, done, *_ = env.step(action)
                # print(state, action, reward, next_state)
                self.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                t += 1
            self.episode_rewards.append(total_reward)

    def get_policy(self) -> np.ndarray:
        """
        Get the greedy policy
        """
        return np.argmax(self.Q, axis=1)

    def play(self, env: "gym.Env") -> None:
        """
        Play the game using the greedy policy
        """
        state, _ = env.reset()
        done = False
        while not done:
            action = self.get_policy()[state]
            state, _, done, *_ = env.step(action)
            env.render()

    def evaluate(
        self, env: "gym.Env", n_episodes: int = 100, max_steps: int = 1000
    ) -> float:
        """
        Evaluate the greedy policy
        """
        total_rewards = []
        for episode in range(n_episodes):
            t = 0
            state, _ = env.reset()
            done = False
            total_reward = 0
            while t < max_steps and not done:
                action = self.get_policy()[state]
                state, reward, done, *_ = env.step(action)
                total_reward += reward
                t += 1
            total_rewards.append(total_reward)
        return np.mean(total_rewards), np.std(total_rewards), np.min(total_rewards), np.max(total_rewards)


# Test
if __name__ == "__main__":

    import gym

    env = gym.make(
        "Taxi-v3"
        #    , render_mode="human"
    )

    env.action_space.seed(42)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    qlearning = QLearning(n_states, n_actions, gamma=0.9, alpha=1, epsilon=0.1)
    qlearning.solve(env, n_episodes=1000, max_steps=1000)
    print(qlearning.Q)
    print(qlearning.get_policy())
    import matplotlib.pyplot as plt

    plt.plot(qlearning.episode_rewards)
    plt.show()

    env.close()

    env = gym.make("Taxi-v3", render_mode="human")
    qlearning.play(env)

    env.close()
