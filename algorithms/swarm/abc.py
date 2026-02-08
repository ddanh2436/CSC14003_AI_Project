import numpy as np
from algorithms.optimizer import Optimizer

class ArtificialBeeColony(Optimizer):
    def __init__(self, problem, pop_size=50, limit=100, **kwargs):
        super().__init__(problem, pop_size=pop_size, **kwargs)
        self.n_food = pop_size // 2 
        self.limit = limit

    def _evolve(self):
        dim = self.problem.dim
        lb, ub = self.problem.bounds[:, 0], self.problem.bounds[:, 1]
        
        # 1. Khởi tạo
        pop = np.random.uniform(lb, ub, (self.n_food, dim))
        fitness = np.apply_along_axis(self.problem.fitness, 1, pop)
        trials = np.zeros(self.n_food)
        
        best_idx = np.argmin(fitness)
        self.update_global_best(pop[best_idx], fitness[best_idx])
        self.save_history()

        for _ in range(self.max_iter):
            # --- GIAI ĐOẠN EMPLOYED BEES (Vectorized) ---
            # Chọn k ngẫu nhiên cho mỗi i (k != i)
            k_indices = np.array([np.random.choice(np.delete(np.arange(self.n_food), i)) for i in range(self.n_food)])
            phi = np.random.uniform(-1, 1, (self.n_food, dim))
            
            # Tạo ứng viên mới cho toàn bộ quần thể
            new_pop = np.clip(pop + phi * (pop - pop[k_indices]), lb, ub)
            new_fitness = np.apply_along_axis(self.problem.fitness, 1, new_pop)
            
            # Cập nhật bằng Greedy Selection (Masking)
            better_mask = new_fitness < fitness
            pop[better_mask] = new_pop[better_mask]
            fitness[better_mask] = new_fitness[better_mask]
            trials[better_mask] = 0
            trials[~better_mask] += 1

            # --- GIAI ĐOẠN ONLOOKER BEES (Vectorized) ---
            # Tính xác suất dựa trên fitness (càng nhỏ càng tốt)
            fit_inv = 1.0 / (1.0 + fitness + np.abs(np.min(fitness)))
            probs = fit_inv / np.sum(fit_inv)
            
            # Chọn n_food nguồn thức ăn dựa trên xác suất (Onlookers chọn)
            selected_idx = np.random.choice(np.arange(self.n_food), size=self.n_food, p=probs)
            
            # Lại chọn k ngẫu nhiên cho các nguồn đã chọn
            k_indices_onlooker = np.array([np.random.choice(np.delete(np.arange(self.n_food), i)) for i in selected_idx])
            phi_onlooker = np.random.uniform(-1, 1, (self.n_food, dim))
            
            new_pop_on = np.clip(pop[selected_idx] + phi_onlooker * (pop[selected_idx] - pop[k_indices_onlooker]), lb, ub)
            new_fit_on = np.apply_along_axis(self.problem.fitness, 1, new_pop_on)
            
            # Cập nhật cho Onlooker
            for idx, i_source in enumerate(selected_idx):
                if new_fit_on[idx] < fitness[i_source]:
                    pop[i_source] = new_pop_on[idx]
                    fitness[i_source] = new_fit_on[idx]
                    trials[i_source] = 0
                else:
                    trials[i_source] += 1

            # --- GIAI ĐOẠN SCOUT BEES ---
            scout_idx = np.where(trials > self.limit)[0]
            if len(scout_idx) > 0:
                pop[scout_idx] = np.random.uniform(lb, ub, (len(scout_idx), dim))
                fitness[scout_idx] = np.apply_along_axis(self.problem.fitness, 1, pop[scout_idx])
                trials[scout_idx] = 0

            # Cập nhật Best toàn cục
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.global_best_fitness:
                self.update_global_best(pop[min_idx], fitness[min_idx])
            
            self.save_history()

        return self.global_best_solution, self.global_best_fitness