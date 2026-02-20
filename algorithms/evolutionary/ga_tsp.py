import numpy as np
from algorithms.optimizer import Optimizer

class GeneticAlgorithmTSP(Optimizer):
    def __init__(self, problem, pop_size=50, mutation_rate=0.1, crossover_rate=0.9, **kwargs):
        super().__init__(problem, pop_size=pop_size, **kwargs)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def _evolve(self):
        n_cities = self.problem.n_cities
        # Khởi tạo quần thể hoán vị ngẫu nhiên
        pop = np.array([np.random.permutation(n_cities) for _ in range(self.pop_size)])
        fitness = np.array([self.problem.fitness(ind) for ind in pop])
        
        self.update_global_best(pop[np.argmin(fitness)], np.min(fitness))
        self.save_history()

        for _ in range(self.max_iter):
            # 1. Tournament Selection
            idx1 = np.random.randint(0, self.pop_size, self.pop_size)
            idx2 = np.random.randint(0, self.pop_size, self.pop_size)
            parents = pop[np.where(fitness[idx1] < fitness[idx2], idx1, idx2)].copy()

            offspring = []
            # 2. Crossover: Order Crossover (OX1)
            for i in range(0, self.pop_size, 2):
                p1, p2 = parents[i], parents[(i+1) % self.pop_size]
                if np.random.rand() < self.crossover_rate:
                    # Chọn 2 điểm cắt
                    c1, c2 = sorted(np.random.choice(n_cities, 2, replace=False))
                    
                    # Tạo con 1
                    child1 = np.full(n_cities, -1)
                    child1[c1:c2] = p1[c1:c2] # Giữ nguyên khúc giữa của P1
                    p2_filtered = [x for x in p2 if x not in child1] # Lấy các gen còn lại từ P2
                    child1[child1 == -1] = p2_filtered
                    
                    # Tạo con 2
                    child2 = np.full(n_cities, -1)
                    child2[c1:c2] = p2[c1:c2]
                    p1_filtered = [x for x in p1 if x not in child2]
                    child2[child2 == -1] = p1_filtered
                    
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([p1, p2])
            
            offspring = np.array(offspring)

            # 3. Mutation: Swap Mutation
            for i in range(self.pop_size):
                if np.random.rand() < self.mutation_rate:
                    idx_swap = np.random.choice(n_cities, 2, replace=False)
                    offspring[i, idx_swap[0]], offspring[i, idx_swap[1]] = offspring[i, idx_swap[1]], offspring[i, idx_swap[0]]

            # Đánh giá và cập nhật
            fitness = np.array([self.problem.fitness(ind) for ind in offspring])
            pop = offspring
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.global_best_fitness:
                self.update_global_best(pop[best_idx], fitness[best_idx])
            
            self.save_history()

        return self.global_best_solution, self.global_best_fitness