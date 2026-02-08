import numpy as np
from algorithms.optimizer import Optimizer

class EvolutionStrategy(Optimizer):
    """
    Phiên bản (1 + lambda) Evolution Strategy đơn giản.
    Sử dụng đột biến Gaussian với độ lệch chuẩn tự thích nghi (Step size).
    """
    def __init__(self, problem, sigma=0.1, lr=0.1, **kwargs):
        super().__init__(problem, **kwargs)
        self.sigma = sigma # Bước nhảy ban đầu
        self.lr = lr       # Tốc độ học của sigma (learning rate)

    def _evolve(self):
        dim = self.problem.dim
        lb, ub = self.problem.bounds[:, 0], self.problem.bounds[:, 1]
        
        # Khởi tạo một cá thể duy nhất (Parent)
        parent = np.random.uniform(lb, ub, dim)
        parent_fitness = self.problem.fitness(parent)
        sigma = self.sigma
        
        self.update_global_best(parent, parent_fitness)
        self.save_history()

        for _ in range(self.max_iter):
            # Tạo các con (Offspring) - số lượng bằng pop_size
            offspring_list = []
            for _ in range(self.pop_size):
                noise = np.random.normal(0, sigma, dim)
                child = np.clip(parent + noise, lb, ub)
                child_fitness = self.problem.fitness(child)
                offspring_list.append((child, child_fitness))
            
            # Tìm con tốt nhất trong đám con
            offspring_list.sort(key=lambda x: x[1])
            best_child, best_child_fitness = offspring_list[0]
            
            # Rule 1/5 of Success (Cơ chế đơn giản để điều chỉnh sigma)
            # Nếu con tốt hơn cha, giữ lại con và tăng bước nhảy
            if best_child_fitness < parent_fitness:
                parent, parent_fitness = best_child, best_child_fitness
                sigma *= (1 + self.lr)
                self.update_global_best(parent, parent_fitness)
            else:
                # Nếu không cải thiện, giảm bước nhảy để hội tụ kỹ hơn
                sigma *= (1 - self.lr)
            
            self.save_history()

        return self.global_best_solution, self.global_best_fitness