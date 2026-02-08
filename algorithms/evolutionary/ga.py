import numpy as np
from algorithms.optimizer import Optimizer

class GeneticAlgorithm(Optimizer):
    def __init__(self, problem, pop_size=50, mutation_rate=0.1, crossover_rate=0.9, **kwargs):
        super().__init__(problem, pop_size=pop_size, **kwargs)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def _evolve(self):
        dim = self.problem.dim
        lb, ub = self.problem.bounds[:, 0], self.problem.bounds[:, 1]
        
        pop = np.random.uniform(lb, ub, (self.pop_size, dim))
        fitness = np.apply_along_axis(self.problem.fitness, 1, pop)
        
        best_idx = np.argmin(fitness)
        self.update_global_best(pop[best_idx], fitness[best_idx])
        self.save_history()

        for _ in range(self.max_iter):
            # A. Selection (Tournament 2 - Vectorized)
            idx1 = np.random.randint(0, self.pop_size, self.pop_size)
            idx2 = np.random.randint(0, self.pop_size, self.pop_size)
            parents = np.where((fitness[idx1] < fitness[idx2])[:, np.newaxis], pop[idx1], pop[idx2])

            # B. Crossover (Uniform Crossover - Vectorized)
            # Trộn ngẫu nhiên các cặp bố mẹ
            parents_shuffled = parents[np.random.permutation(self.pop_size)]
            cross_mask = (np.random.rand(self.pop_size, dim) < 0.5) & (np.random.rand(self.pop_size, 1) < self.crossover_rate)
            offspring = np.where(cross_mask, parents, parents_shuffled)

            # C. Mutation (Gaussian Mutation - Vectorized)
            mutate_mask = np.random.rand(self.pop_size, dim) < self.mutation_rate
            # Độ lệch chuẩn của đột biến giảm dần theo thời gian để ổn định (Tùy chọn)
            noise = np.random.normal(0, 0.5, (self.pop_size, dim)) 
            offspring[mutate_mask] += noise[mutate_mask]
            offspring = np.clip(offspring, lb, ub)

            # D. Evaluation & Update
            offspring_fitness = np.apply_along_axis(self.problem.fitness, 1, offspring)
            
            # Elitism: Giữ lại những cá thể tốt nhất từ cả bố mẹ và con (Tùy chọn để tăng độ ổn định)
            # Ở đây ta dùng Simple GA: Thế hệ con thay thế hoàn toàn thế hệ cha
            pop = offspring
            fitness = offspring_fitness
            
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.global_best_fitness:
                self.update_global_best(pop[current_best_idx], fitness[current_best_idx])
            
            self.save_history()

        return self.global_best_solution, self.global_best_fitness