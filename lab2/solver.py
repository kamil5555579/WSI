import numpy as np

class EvolutionaryAlgorithm:
    def __init__(
        self,
        f: callable,
        P0: np.ndarray,
        mu: int,
        sigma: float,
        max_iter: int = 1000,
        eps: float = 1e-3,
    ):
        """
        :param f: function to minimize
        :param P0: initial population
        :param mu: population size
        :param sigma: mutation step
        :param max_iter: maximum number of iterations
        :param eps: threshold for stopping criterion
        """
        self.f = f
        self.P0 = P0
        self.mu = mu
        self.sigma = sigma
        self.max_iter = max_iter
        self.eps = eps

    def get_fitness(self, P: np.ndarray):
        """
        :param P: population
        :return: fitness values
        """
        return 1 / self.f(P)

    def get_best(self, P: np.ndarray, f_P: np.ndarray):
        """
        :param P: population
        :param f_P: fitness values
        :return: best individual and its fitness value
        """
        return P[np.argmax(f_P)], np.max(f_P)

    def get_n_best(self, P: np.ndarray, f_P: np.ndarray, n: int):
        """
        :param P: population
        :param f_P: fitness values
        :param n: number of best individuals to return
        :return: n best individuals and their fitness values
        """
        idx = np.argsort(f_P)
        return P[idx[-n:]], f_P[idx[-n:]]

    def reproduce_roulette(self, P: np.ndarray, f_P: np.ndarray):
        """
        :param P: population
        :param f_P: fitness values
        :return: new population
        """
        idx = np.random.choice(
            np.arange(self.mu), size=self.mu, p=f_P / f_P.sum(), replace=True
        )
        return P[idx]

    def reproduce_tournament(
        self, P: np.ndarray, f_P: np.ndarray, tournament_size: int
    ):
        """
        :param P: population
        :param f_P: fitness values
        :param tournament_size: number of individuals in a tournament
        :return: new population
        """

        P_new = np.zeros_like(P)
        for i in range(self.mu):
            idx = np.random.choice(
                np.arange(self.mu), size=tournament_size, replace=True
            )
            P_new[i] = P[np.argmax(f_P[idx])]
        return P_new

    def mutate(self, P: np.ndarray):
        """
        :param P: population
        :return: mutated population
        """
        return P + self.sigma * np.random.normal(0, 1, size=P.shape)

    def successions(self, P: np.ndarray, f_P: np.ndarray, elite: np.ndarray):
        """
        :param P: population
        :param f_P: fitness values
        :return: new population
        """
        P_new = np.zeros_like(P)
        P_new[0] = elite
        P_new[1:] = self.get_n_best(P, f_P, self.mu - 1)[0]

        return P_new

    def solve(self, method="roulette"):
        """
        :param method: reproduction method, either "roulette" or "tournament"
        :return: population over iterations, fitness values over iterations
        """
        P = self.P0
        f_P = self.get_fitness(P)
        x_0, f_0 = self.get_best(P, f_P)
        x_t = [x_0]
        f_t = [1 / f_0]
        for t in range(self.max_iter):
            elite = self.get_best(P, f_P)[0]
            if method == "roulette":
                R = self.reproduce_roulette(P, f_P)
            elif method == "tournament":
                R = self.reproduce_tournament(P, f_P, 2)
            else:
                raise ValueError("Unknown method")
            M = self.mutate(R)
            f_M = self.get_fitness(M)
            P = self.successions(M, f_M, elite)
            f_P = self.get_fitness(P)
            x_t.append(self.get_best(P, f_P)[0])
            f_t.append(1 / (self.get_best(P, f_P)[1]))

            if len(f_t) > 200:
                if f_t[-1] == f_t[-200]:
                    break

        return x_t, f_t
