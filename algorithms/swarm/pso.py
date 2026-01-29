import numpy as np
from algorithms.optimizer import Optimizer

class ParticleSwarmOptimization(Optimizer):
    def __init__(self, problem, pop_size=30, w=0.7, c1=1.5, c2=1.5, **kwargs):
        """
        w: Inertia weight (Quán tính)
        c1: Cognitive weight (Hệ số cá nhân)
        c2: Social weight (Hệ số xã hội)
        """
        super().__init__(problem, pop_size=pop_size, **kwargs)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def _evolve(self):
        # 1. Khởi tạo
        dim = self.problem.dim
        lb = self.problem.bounds[:, 0]
        ub = self.problem.bounds[:, 1]
        
        # Vị trí và vận tốc
        X = np.random.uniform(lb, ub, (self.pop_size, dim))
        V = np.random.uniform(-1, 1, (self.pop_size, dim))
        
        # P_best (Cá nhân tốt nhất)
        P_best = X.copy()
        P_best_val = np.apply_along_axis(self.problem.fitness, 1, X)
        
        # Cập nhật Global Best lần đầu
        min_idx = np.argmin(P_best_val)
        self.update_global_best(P_best[min_idx], P_best_val[min_idx])
        self.save_history()

        # 2. Vòng lặp
        for _ in range(self.max_iter):
            r1 = np.random.rand(self.pop_size, dim)
            r2 = np.random.rand(self.pop_size, dim)
            
            # Cập nhật vận tốc
            # v_new = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
            V = (self.w * V + 
                 self.c1 * r1 * (P_best - X) + 
                 self.c2 * r2 * (self.global_best_solution - X))
            
            # Cập nhật vị trí
            X = X + V
            X = np.clip(X, lb, ub) # Giữ trong biên
            
            # Đánh giá
            current_vals = np.apply_along_axis(self.problem.fitness, 1, X)
            
            # Cập nhật P_best
            better_mask = current_vals < P_best_val
            P_best[better_mask] = X[better_mask]
            P_best_val[better_mask] = current_vals[better_mask]
            
            # Cập nhật Global Best
            min_val = np.min(P_best_val)
            min_idx = np.argmin(P_best_val)
            
            if min_val < self.global_best_fitness:
                self.update_global_best(P_best[min_idx], min_val)
                
            self.save_history()
            
        return self.global_best_solution, self.global_best_fitness