import numpy as np
from algorithms.optimizer import Optimizer

class ArtificialBeeColony(Optimizer):
    def __init__(self, problem, pop_size=50, limit=50, **kwargs):
        # pop_size trong ABC thường là tổng số ong (Employed + Onlooker)
        # Số lượng nguồn thức ăn (SN) = pop_size / 2
        super().__init__(problem, pop_size=pop_size, **kwargs)
        self.n_food = pop_size // 2 
        self.limit = limit

    def _evolve(self):
        dim = self.problem.dim
        lb = self.problem.bounds[:, 0]
        ub = self.problem.bounds[:, 1]
        
        # Khởi tạo nguồn thức ăn (Employed bees ban đầu)
        pop = np.random.uniform(lb, ub, (self.n_food, dim))
        fitness = np.apply_along_axis(self.problem.fitness, 1, pop)
        trials = np.zeros(self.n_food) # Đếm số lần không cải thiện
        
        # Cập nhật Global Best
        best_idx = np.argmin(fitness)
        self.update_global_best(pop[best_idx], fitness[best_idx])
        self.save_history()

        def mutate(i):
            k = i
            while k == i:
                k = np.random.randint(0, self.n_food)
            
            phi = np.random.uniform(-1, 1, dim)
            new_sol = pop[i] + phi * (pop[i] - pop[k])
            new_sol = np.clip(new_sol, lb, ub)
            new_fit = self.problem.fitness(new_sol)
            
            if new_fit < fitness[i]:
                pop[i] = new_sol
                fitness[i] = new_fit
                trials[i] = 0
            else:
                trials[i] += 1

        for _ in range(self.max_iter):
            # 1. Employed Bees Phase
            for i in range(self.n_food):
                mutate(i)

            # 2. Onlooker Bees Phase
            # Tính xác suất (Roulette Wheel)
            # Chuyển fitness (min problem) sang xác suất: fit càng nhỏ prob càng to
            fit_inv = 1.0 / (1.0 + fitness + abs(np.min(fitness))) 
            probs = fit_inv / np.sum(fit_inv)
            
            for _ in range(self.n_food):
                # Chọn nguồn thức ăn để khai thác
                i = np.searchsorted(np.cumsum(probs), np.random.rand())
                i = min(i, self.n_food - 1)
                mutate(i)

            # 3. Scout Bees Phase
            # Tìm nguồn thức ăn đã cạn kiệt (vượt quá limit)
            max_trials_idx = np.argmax(trials)
            if trials[max_trials_idx] > self.limit:
                pop[max_trials_idx] = np.random.uniform(lb, ub, dim)
                fitness[max_trials_idx] = self.problem.fitness(pop[max_trials_idx])
                trials[max_trials_idx] = 0

            # Cập nhật kết quả tốt nhất vòng này
            curr_best_idx = np.argmin(fitness)
            if fitness[curr_best_idx] < self.global_best_fitness:
                self.update_global_best(pop[curr_best_idx], fitness[curr_best_idx])
            
            self.save_history()

        return self.global_best_solution, self.global_best_fitness