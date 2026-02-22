import numpy as np
from algorithms.optimizer import Optimizer

class SimulatedAnnealingTSP(Optimizer):
    def __init__(self, problem, initial_temp=1000, cooling_rate=0.99, **kwargs):
        super().__init__(problem, **kwargs)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def _evolve(self):
        current_sol = np.random.permutation(self.problem.n_cities)
        current_fit = self.problem.fitness(current_sol)
        self.update_global_best(current_sol, current_fit)
        
        temp = self.initial_temp

        for _ in range(self.max_iter):
            # 2-opt neighborhood: Chọn 2 điểm và đảo ngược đoạn ở giữa
            i, j = sorted(np.random.choice(self.problem.n_cities, 2, replace=False))
            neighbor = current_sol.copy()
            neighbor[i:j] = neighbor[i:j][::-1] # Phép đảo ngược (Reverse)
            
            neighbor_fit = self.problem.fitness(neighbor)
            delta = neighbor_fit - current_fit

            if delta < 0 or np.random.rand() < np.exp(-delta / temp):
                current_sol = neighbor
                current_fit = neighbor_fit
                if current_fit < self.global_best_fitness:
                    self.update_global_best(current_sol, current_fit)

            temp *= self.cooling_rate
            self.save_history()

        return self.global_best_solution, self.global_best_fitness