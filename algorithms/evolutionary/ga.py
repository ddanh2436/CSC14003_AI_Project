import numpy as np
from algorithms.optimizer import Optimizer

class GeneticAlgorithm(Optimizer):
    def __init__(self, problem, pop_size=50, mutation_rate=0.1, crossover_rate=0.9, **kwargs):
        super().__init__(problem, pop_size=pop_size, **kwargs)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def _evolve(self):
        # 1. Khởi tạo quần thể
        pop = np.random.uniform(
            self.problem.bounds[:, 0], self.problem.bounds[:, 1], 
            (self.pop_size, self.problem.dim)
        )
        fitness = np.apply_along_axis(self.problem.fitness, 1, pop)
        
        # Cập nhật best ban đầu
        best_idx = np.argmin(fitness)
        self.update_global_best(pop[best_idx], fitness[best_idx])
        
        # --- SỬA LỖI Ở ĐÂY: Xóa tham số truyền vào ---
        self.save_history() 

        # 2. Vòng lặp tiến hóa
        for _ in range(self.max_iter):
            # A. Selection (Tournament)
            idx1 = np.random.randint(0, self.pop_size, self.pop_size)
            idx2 = np.random.randint(0, self.pop_size, self.pop_size)
            mask = fitness[idx1] < fitness[idx2]
            parents = pop[np.where(mask, idx1, idx2)]

            # B. Crossover
            parents2 = parents.copy()
            np.random.shuffle(parents2)
            cross_mask = np.random.rand(self.pop_size, self.problem.dim) < 0.5
            perform_cross = np.random.rand(self.pop_size, 1) < self.crossover_rate
            offspring = np.where(cross_mask & perform_cross, parents, parents2)
            offspring = np.where(perform_cross, offspring, parents)

            # C. Mutation
            mutation_noise = np.random.normal(0, 1.0, size=offspring.shape)
            mutate_mask = np.random.rand(self.pop_size, self.problem.dim) < self.mutation_rate
            offspring[mutate_mask] += mutation_noise[mutate_mask]
            offspring = np.clip(offspring, self.problem.bounds[:, 0], self.problem.bounds[:, 1])

            # D. Update
            offspring_fitness = np.apply_along_axis(self.problem.fitness, 1, offspring)
            pop = offspring
            fitness = offspring_fitness
            
            # Cập nhật Global Best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.global_best_fitness:
                self.update_global_best(pop[current_best_idx], fitness[current_best_idx])
            
            # --- SỬA LỖI Ở ĐÂY: Xóa tham số truyền vào ---
            self.save_history()

        return self.global_best_solution, self.global_best_fitness