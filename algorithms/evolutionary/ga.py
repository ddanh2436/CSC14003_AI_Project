import numpy as np
from algorithms.optimizer import Optimizer

class GeneticAlgorithm(Optimizer):
    """
    Genetic Algorithm (GA) - Tối ưu hóa cho không gian liên tục (Continuous).
    Sử dụng:
    - Chọn lọc: Tournament Selection.
    - Lai ghép: Uniform Crossover.
    - Đột biến: Gaussian Mutation.
    - Chiến lược: Simple GA kết hợp Elitism (Giữ lại tinh hoa).
    """
    def __init__(self, problem, pop_size=50, mutation_rate=0.1, crossover_rate=0.9, **kwargs):
        super().__init__(problem, pop_size=pop_size, **kwargs)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def _evolve(self):
        dim = self.problem.dim
        
        # Xử lý linh hoạt: Nếu bài toán không có bounds (như Knapsack), mặc định là [0, 1]
        if self.problem.bounds is not None:
            lb = self.problem.bounds[:, 0]
            ub = self.problem.bounds[:, 1]
        else:
            lb = np.zeros(dim)
            ub = np.ones(dim)

        # 1. Khởi tạo quần thể ngẫu nhiên
        pop = np.random.uniform(lb, ub, (self.pop_size, dim))
        fitness = np.apply_along_axis(self.problem.fitness, 1, pop)
        
        # Cập nhật kết quả tốt nhất ban đầu
        best_idx = np.argmin(fitness)
        self.update_global_best(pop[best_idx], fitness[best_idx])
        
        # TỐI ƯU HÓA: Bỏ truyền tham số `pop` để không tính Diversity, giúp tăng tốc tối đa
        self.save_history() 

        # 2. Vòng lặp tiến hóa
        for _ in range(self.max_iter):
            # --- A. SELECTION (Tournament) ---
            idx1 = np.random.randint(0, self.pop_size, self.pop_size)
            idx2 = np.random.randint(0, self.pop_size, self.pop_size)
            
            # Chọn cá thể có fitness tốt hơn (nhỏ hơn) làm cha mẹ
            mask_tournament = fitness[idx1] < fitness[idx2]
            parents = np.where(mask_tournament[:, np.newaxis], pop[idx1], pop[idx2])

            # --- B. CROSSOVER (Uniform) ---
            parents_shuffled = parents.copy()
            np.random.shuffle(parents_shuffled) # Trộn ngẫu nhiên để lai ghép
            
            # Mặt nạ lai ghép: Xác suất lấy gen từ bố hoặc mẹ
            cross_mask = np.random.rand(self.pop_size, dim) < 0.5
            perform_cross = np.random.rand(self.pop_size, 1) < self.crossover_rate
            
            offspring = np.where(cross_mask & perform_cross, parents, parents_shuffled)
            offspring = np.where(perform_cross, offspring, parents)

            # --- C. MUTATION (Gaussian) ---
            mut_mask = np.random.rand(self.pop_size, dim) < self.mutation_rate
            
            # Nhiễu Gaussian với độ phân tán là 10% của không gian tìm kiếm
            sigma = 0.1 * (ub - lb)
            noise = np.random.normal(0, sigma, size=(self.pop_size, dim))
            
            offspring[mut_mask] += noise[mut_mask]
            
            # Cắt xén (Clip) để đảm bảo không có gen nào văng ra ngoài không gian
            offspring = np.clip(offspring, lb, ub)

            # --- D. ĐÁNH GIÁ LẠI & ELITISM ---
            offspring_fitness = np.apply_along_axis(self.problem.fitness, 1, offspring)
            
            # Tìm cá thể xuất sắc nhất đời cha và cá thể kém nhất đời con
            best_parent_idx = np.argmin(fitness)
            worst_offspring_idx = np.argmax(offspring_fitness)
            
            pop = offspring.copy()
            fitness = offspring_fitness.copy()
            
            # Elitism: Đảm bảo cá thể tốt nhất đời trước không bị mất đi do lai/đột biến
            pop[worst_offspring_idx] = parents[best_parent_idx]
            fitness[worst_offspring_idx] = self.problem.fitness(parents[best_parent_idx])
            
            # Cập nhật kết quả tốt nhất toàn cục
            curr_best_idx = np.argmin(fitness)
            if fitness[curr_best_idx] < self.global_best_fitness:
                self.update_global_best(pop[curr_best_idx], fitness[curr_best_idx])
            
            # TỐI ƯU HÓA: Bỏ truyền tham số `pop` để không tính Diversity
            self.save_history()

        return self.global_best_solution, self.global_best_fitness