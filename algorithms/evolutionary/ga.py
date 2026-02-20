import numpy as np
from algorithms.optimizer import Optimizer

class GeneticAlgorithm(Optimizer):
    def __init__(self, problem, pop_size=50, mutation_rate=0.1, crossover_rate=0.9, **kwargs):
        super().__init__(problem, pop_size=pop_size, **kwargs)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def _initialize_population(self):
        """Tự động phát hiện kiểu bài toán để khởi tạo"""
        # 1. Nếu là TSP -> Sinh hoán vị
        if hasattr(self.problem, 'n_cities'):
            pop = []
            for _ in range(self.pop_size):
                pop.append(np.random.permutation(self.problem.n_cities))
            return np.array(pop)
        
        # 2. Nếu là Knapsack -> Sinh nhị phân 0/1
        elif hasattr(self.problem, 'n_items'):
            return np.random.randint(0, 2, (self.pop_size, self.problem.dim))
            
        # 3. Mặc định: Continuous -> Sinh số thực trong bounds
        else:
            return np.random.uniform(
                self.problem.bounds[:, 0], self.problem.bounds[:, 1], 
                (self.pop_size, self.problem.dim)
            )

    def _evolve(self):
        # 1. Khởi tạo quần thể thông minh
        pop = self._initialize_population()
        
        # Tính fitness ban đầu
        fitness = np.array([self.problem.fitness(ind) for ind in pop])
        
        # Cập nhật Global Best
        best_idx = np.argmin(fitness)
        self.update_global_best(pop[best_idx], fitness[best_idx])
        self.save_history()

        # 2. Vòng lặp tiến hóa
        for _ in range(self.max_iter):
            # A. Selection (Tournament)
            idx1 = np.random.randint(0, self.pop_size, self.pop_size)
            idx2 = np.random.randint(0, self.pop_size, self.pop_size)
            mask = fitness[idx1] < fitness[idx2]
            parents = pop[np.where(mask, idx1, idx2)].copy()

            offspring = parents.copy()
            
            # --- TRƯỜNG HỢP 1: TSP (Rời rạc - Hoán vị) ---
            if hasattr(self.problem, 'n_cities'):
                for i in range(self.pop_size):
                    if np.random.rand() < self.mutation_rate:
                        p1, p2 = np.random.choice(self.problem.n_cities, 2, replace=False)
                        offspring[i][p1], offspring[i][p2] = offspring[i][p2], offspring[i][p1]

            # --- TRƯỜNG HỢP 2: KNAPSACK (Rời rạc - Nhị phân) ---
            elif hasattr(self.problem, 'n_items'):
                # Mutation: Đảo Bit (Flip Bit)
                mask_mut = np.random.rand(self.pop_size, self.problem.dim) < self.mutation_rate
                offspring[mask_mut] = 1 - offspring[mask_mut]

            # --- TRƯỜNG HỢP 3: CONTINUOUS (Liên tục) ---
            else:
                # Crossover
                parents2 = parents.copy()
                np.random.shuffle(parents2)
                cross_mask = np.random.rand(self.pop_size, self.problem.dim) < 0.5
                perform_cross = np.random.rand(self.pop_size, 1) < self.crossover_rate
                offspring = np.where(cross_mask & perform_cross, parents, parents2)
                offspring = np.where(perform_cross, offspring, parents)
                
                # Mutation
                mut_mask = np.random.rand(self.pop_size, self.problem.dim) < self.mutation_rate
                noise = np.random.normal(0, 1.0, size=offspring.shape)
                offspring[mut_mask] += noise[mut_mask]
                offspring = np.clip(offspring, self.problem.bounds[:, 0], self.problem.bounds[:, 1])

            # C. Đánh giá lại
            offspring_fitness = np.array([self.problem.fitness(ind) for ind in offspring])
            
            pop = offspring
            fitness = offspring_fitness
            
            curr_best_idx = np.argmin(fitness)
            if fitness[curr_best_idx] < self.global_best_fitness:
                self.update_global_best(pop[curr_best_idx], fitness[curr_best_idx])
            
            self.save_history()

        return self.global_best_solution, self.global_best_fitness