import numpy as np
from algorithms.optimizer import Optimizer

class DifferentialEvolution(Optimizer):
    def __init__(self, problem, pop_size=50, F=0.8, CR=0.9, **kwargs):
        super().__init__(problem, pop_size=pop_size, **kwargs)
        self.F = F
        self.CR = CR

    def _evolve(self):
        dim = self.problem.dim
        lb, ub = self.problem.bounds[:, 0], self.problem.bounds[:, 1]
        
        # 1. Khởi tạo
        pop = np.random.uniform(lb, ub, (self.pop_size, dim))
        fitness = np.apply_along_axis(self.problem.fitness, 1, pop)
        
        best_idx = np.argmin(fitness)
        self.update_global_best(pop[best_idx], fitness[best_idx])
        self.save_history()

        for _ in range(self.max_iter):
            # 2. Đột biến (Đã tối ưu hóa, loại bỏ np.delete để tăng tốc)
            idxs = np.zeros((self.pop_size, 3), dtype=int)
            for i in range(self.pop_size):
                candidates = [j for j in range(self.pop_size) if j != i]
                idxs[i] = np.random.choice(candidates, 3, replace=False)
                
            a, b, c = pop[idxs[:, 0]], pop[idxs[:, 1]], pop[idxs[:, 2]]
            mutants = np.clip(a + self.F * (b - c), lb, ub)
            
            # 3. Lai ghép vector hóa
            cross_points = np.random.rand(self.pop_size, dim) < self.CR
            j_rand = np.random.randint(0, dim, self.pop_size)
            cross_points[np.arange(self.pop_size), j_rand] = True
            
            trials = np.where(cross_points, mutants, pop)
            trial_fitness = np.apply_along_axis(self.problem.fitness, 1, trials)
            
            # 4. Chọn lọc vector hóa
            better_mask = trial_fitness < fitness
            pop[better_mask] = trials[better_mask]
            fitness[better_mask] = trial_fitness[better_mask] 
            
            # Cập nhật Global Best
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.global_best_fitness:
                self.update_global_best(pop[min_idx], fitness[min_idx])
            
            self.save_history()

        return self.global_best_solution, self.global_best_fitness