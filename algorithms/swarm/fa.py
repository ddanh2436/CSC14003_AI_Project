import numpy as np
from algorithms.optimizer import Optimizer

class FireflyAlgorithm(Optimizer):
    def __init__(self, problem, pop_size=40, beta0=1.0, gamma=1.0, alpha=0.2, **kwargs):
        super().__init__(problem, pop_size=pop_size, **kwargs)
        self.beta0 = beta0  # Attractiveness at r=0
        self.gamma = gamma  # Absorption coefficient
        self.alpha = alpha  # Randomization parameter

    def _evolve(self):
        dim = self.problem.dim
        lb = self.problem.bounds[:, 0]
        ub = self.problem.bounds[:, 1]
        
        # Khởi tạo
        X = np.random.uniform(lb, ub, (self.pop_size, dim))
        Light = np.apply_along_axis(self.problem.fitness, 1, X)
        
        # Update Best
        min_idx = np.argmin(Light)
        self.update_global_best(X[min_idx], Light[min_idx])
        self.save_history()

        for _ in range(self.max_iter):
            # So sánh từng cặp đom đóm
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    # Nếu j sáng hơn i (fitness nhỏ hơn) -> i bay về phía j
                    if Light[j] < Light[i]:
                        r = np.linalg.norm(X[i] - X[j])
                        beta = self.beta0 * np.exp(-self.gamma * r**2)
                        
                        noise = self.alpha * (np.random.rand(dim) - 0.5)
                        X[i] += beta * (X[j] - X[i]) + noise
                        X[i] = np.clip(X[i], lb, ub)
                        
                        Light[i] = self.problem.fitness(X[i])
            
            # Giảm alpha dần để ổn định
            self.alpha *= 0.98
            
            # Cập nhật Best
            curr_best_val = np.min(Light)
            curr_best_idx = np.argmin(Light)
            
            if curr_best_val < self.global_best_fitness:
                self.update_global_best(X[curr_best_idx], curr_best_val)
                
            self.save_history()

        return self.global_best_solution, self.global_best_fitness