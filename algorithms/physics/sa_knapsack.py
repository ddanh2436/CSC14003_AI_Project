import numpy as np
from algorithms.optimizer import Optimizer

class SimulatedAnnealingKnapsack(Optimizer):
    def __init__(self, problem, initial_temp=1000, cooling_rate=0.95, **kwargs):
        super().__init__(problem, **kwargs)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def _evolve(self):
        current_sol = np.random.randint(0, 2, self.problem.n_items)
        current_fit = self.problem.fitness(current_sol)
        self.update_global_best(current_sol, current_fit)
        
        temp = self.initial_temp

        for _ in range(self.max_iter):
            # Đảo bit để tạo hàng xóm
            neighbor = current_sol.copy()
            idx = np.random.randint(0, self.problem.n_items)
            neighbor[idx] = 1 - neighbor[idx]
            
            neighbor_fit = self.problem.fitness(neighbor)
            delta = neighbor_fit - current_fit

            # Chấp nhận cái tốt hơn HOẶC chấp nhận cái xấu theo xác suất
            if delta < 0 or np.random.rand() < np.exp(-delta / temp):
                current_sol = neighbor
                current_fit = neighbor_fit
                if current_fit < self.global_best_fitness:
                    self.update_global_best(current_sol, current_fit)

            temp *= self.cooling_rate
            self.save_history()

        return self.global_best_solution, self.global_best_fitness