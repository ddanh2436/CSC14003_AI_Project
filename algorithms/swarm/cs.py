import numpy as np
import math
from algorithms.optimizer import Optimizer

class CuckooSearch(Optimizer):
    def __init__(self, problem, pop_size=25, pa=0.25, **kwargs):
        super().__init__(problem, pop_size=pop_size, **kwargs)
        self.pa = pa # Discovery rate (Xác suất bị phát hiện)

    def _levy_flight(self, beta=1.5):
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.problem.dim)
        v = np.random.normal(0, 1, self.problem.dim)
        step = u / abs(v) ** (1 / beta)
        return step

    def _evolve(self):
        dim = self.problem.dim
        lb = self.problem.bounds[:, 0]
        ub = self.problem.bounds[:, 1]
        
        # Khởi tạo tổ chim
        nests = np.random.uniform(lb, ub, (self.pop_size, dim))
        fitness = np.apply_along_axis(self.problem.fitness, 1, nests)
        
        # Best ban đầu
        best_idx = np.argmin(fitness)
        self.update_global_best(nests[best_idx], fitness[best_idx])
        self.save_history()

        for _ in range(self.max_iter):
            # 1. Tạo cuckoo mới bằng Levy Flight (Global Walk)
            i = np.random.randint(0, self.pop_size)
            step_size = 0.01 * self._levy_flight() * (nests[i] - self.global_best_solution)
            new_cuckoo = nests[i] + step_size * np.random.randn(dim)
            new_cuckoo = np.clip(new_cuckoo, lb, ub)
            new_fit = self.problem.fitness(new_cuckoo)
            
            # Chọn tổ ngẫu nhiên j để đẻ nhờ
            j = np.random.randint(0, self.pop_size)
            if new_fit < fitness[j]:
                nests[j] = new_cuckoo
                fitness[j] = new_fit

            # 2. Loại bỏ tổ xấu (Discovery / Local Walk)
            # Thay thế 1 phần tổ tồi bằng tổ mới
            sorted_idx = np.argsort(fitness)
            n_abandon = int(self.pop_size * self.pa)
            
            # Thay thế n_abandon tổ kém nhất
            for k in range(self.pop_size - n_abandon, self.pop_size):
                idx = sorted_idx[k]
                nests[idx] = np.random.uniform(lb, ub, dim)
                fitness[idx] = self.problem.fitness(nests[idx])

            # Cập nhật Global Best
            curr_best_idx = np.argmin(fitness)
            if fitness[curr_best_idx] < self.global_best_fitness:
                self.update_global_best(nests[curr_best_idx], fitness[curr_best_idx])
            
            self.save_history()

        return self.global_best_solution, self.global_best_fitness