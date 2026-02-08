import numpy as np
from algorithms.optimizer import Optimizer

class ParticleSwarmOptimization(Optimizer):
    def __init__(self, problem, pop_size=30, w=0.7, c1=1.5, c2=1.5, **kwargs):
        super().__init__(problem, pop_size=pop_size, **kwargs)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def _evolve(self):
        dim = self.problem.dim
        lb, ub = self.problem.bounds[:, 0], self.problem.bounds[:, 1]
        
        # Vị trí và vận tốc (Vectorized)
        X = np.random.uniform(lb, ub, (self.pop_size, dim))
        V = np.random.uniform(-1, 1, (self.pop_size, dim))
        
        # Cá nhân tốt nhất (P_best)
        P_best = X.copy()
        P_best_val = np.apply_along_axis(self.problem.fitness, 1, X)
        
        # Global Best
        min_idx = np.argmin(P_best_val)
        self.update_global_best(P_best[min_idx], P_best_val[min_idx])
        self.save_history()

        for _ in range(self.max_iter):
            # Cập nhật vận tốc cho TOÀN BỘ quần thể cùng lúc
            r1 = np.random.rand(self.pop_size, dim)
            r2 = np.random.rand(self.pop_size, dim)
            
            V = (self.w * V + 
                 self.c1 * r1 * (P_best - X) + 
                 self.c2 * r2 * (self.global_best_solution - X))
            
            # Cập nhật vị trí và giới hạn biên
            X = np.clip(X + V, lb, ub)
            
            # Đánh giá fitness đồng loạt
            current_vals = np.apply_along_axis(self.problem.fitness, 1, X)
            
            # Cập nhật P_best bằng mask (Nhanh hơn vòng lặp)
            better_mask = current_vals < P_best_val
            P_best[better_mask] = X[better_mask]
            P_best_val[better_mask] = current_vals[better_mask]
            
            # Cập nhật Global Best
            min_val = np.min(P_best_val)
            if min_val < self.global_best_fitness:
                best_idx = np.argmin(P_best_val)
                self.update_global_best(P_best[best_idx], min_val)
                
            self.save_history()
            
        return self.global_best_solution, self.global_best_fitness