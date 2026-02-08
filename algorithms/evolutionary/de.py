import numpy as np
from algorithms.optimizer import Optimizer

class DifferentialEvolution(Optimizer):
    def __init__(self, problem, pop_size=50, F=0.8, CR=0.9, **kwargs):
        """
        F: Trọng số đột biến (Mutation scale factor) [0, 2]
        CR: Xác suất lai ghép (Crossover probability) [0, 1]
        """
        super().__init__(problem, pop_size=pop_size, **kwargs)
        self.F = F
        self.CR = CR

    def _evolve(self):
        dim = self.problem.dim
        lb, ub = self.problem.bounds[:, 0], self.problem.bounds[:, 1]
        
        # Khởi tạo quần thể
        pop = np.random.uniform(lb, ub, (self.pop_size, dim))
        fitness = np.apply_along_axis(self.problem.fitness, 1, pop)
        
        # Cập nhật Global Best ban đầu
        best_idx = np.argmin(fitness)
        self.update_global_best(pop[best_idx], fitness[best_idx])
        self.save_history()

        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                # 1. Đột biến (Mutation): Chọn 3 cá thể ngẫu nhiên khác i
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                
                # 2. Lai ghép (Crossover): Binomial crossover
                cross_points = np.random.rand(dim) < self.CR
                # Đảm bảo ít nhất một chiều được thay đổi
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = self.problem.fitness(trial)
                
                # 3. Chọn lọc (Selection): So sánh trial với cá thể hiện tại
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
            
            # Cập nhật Global Best sau mỗi vòng lặp
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.global_best_fitness:
                self.update_global_best(pop[current_best_idx], fitness[current_best_idx])
            
            self.save_history()

        return self.global_best_solution, self.global_best_fitness