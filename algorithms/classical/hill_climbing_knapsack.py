import numpy as np
from algorithms.optimizer import Optimizer

class HillClimbingKnapsack(Optimizer):
    def __init__(self, problem, max_iter=1000, **kwargs):
        super().__init__(problem, max_iter=max_iter, **kwargs)

    def _evolve(self):
        # Khởi tạo ngẫu nhiên chuỗi nhị phân (0 hoặc 1)
        current_sol = np.random.randint(0, 2, self.problem.n_items)
        current_fit = self.problem.fitness(current_sol)
        self.update_global_best(current_sol, current_fit)
        self.save_history()

        for _ in range(self.max_iter):
            # TẠO HÀNG XÓM: Đảo ngẫu nhiên 1 bit (0 thành 1, 1 thành 0)
            neighbor = current_sol.copy()
            idx = np.random.randint(0, self.problem.n_items)
            neighbor[idx] = 1 - neighbor[idx] 
            
            neighbor_fit = self.problem.fitness(neighbor)

            # LEO ĐỒI: Chỉ lấy nếu tốt hơn
            if neighbor_fit < current_fit:
                current_sol = neighbor
                current_fit = neighbor_fit
                if current_fit < self.global_best_fitness:
                    self.update_global_best(current_sol, current_fit)
            self.save_history()
            
        return self.global_best_solution, self.global_best_fitness