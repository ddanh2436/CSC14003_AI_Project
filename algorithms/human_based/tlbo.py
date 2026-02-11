import numpy as np
from algorithms.optimizer import Optimizer

class TLBO(Optimizer):
    """
    Teaching-Learning-Based Optimization (TLBO).
    Mô phỏng quá trình dạy và học trong lớp học thông qua hai giai đoạn chính.
    """
    def __init__(self, problem, pop_size=40, **kwargs):
        super().__init__(problem, pop_size=pop_size, **kwargs)

    def _evolve(self):
        dim = self.problem.dim
        lb, ub = self.problem.bounds[:, 0], self.problem.bounds[:, 1]
        
        # 1. Khởi tạo lớp học (Quần thể sinh viên)
        pop = np.random.uniform(lb, ub, (self.pop_size, dim))
        fitness = np.apply_along_axis(self.problem.fitness, 1, pop)
        
        # Cập nhật kết quả tốt nhất ban đầu
        self.update_global_best(pop[np.argmin(fitness)], np.min(fitness))
        self.save_history(pop)

        for _ in range(self.max_iter):
            # --- GIAI ĐOẠN GIẢNG DẠY (TEACHER PHASE) ---
            # Giáo viên là người có kiến thức tốt nhất (Best Fitness)
            teacher = pop[np.argmin(fitness)]
            # Tính kiến thức trung bình của cả lớp
            mean_v = np.mean(pop, axis=0)
            
            # Hệ số giảng dạy TF (Teaching Factor), chọn ngẫu nhiên 1 hoặc 2
            tf = np.random.randint(1, 3, size=self.pop_size)[:, np.newaxis]
            r = np.random.rand(self.pop_size, dim)
            
            # Sinh viên học dựa trên sự khác biệt giữa giáo viên và mặt bằng chung của lớp
            diff_mean = r * (teacher - tf * mean_v)
            new_pop_t = np.clip(pop + diff_mean, lb, ub)
            new_fit_t = np.apply_along_axis(self.problem.fitness, 1, new_pop_t)
            
            # Cập nhật nếu kết quả học tập mới tốt hơn (Greedy Selection)
            better_mask_t = new_fit_t < fitness
            pop[better_mask_t] = new_pop_t[better_mask_t]
            fitness[better_mask_t] = new_fit_t[better_mask_t]

            # --- GIAI ĐOẠN HỌC TẬP (LEARNER PHASE) ---
            # Mỗi sinh viên chọn ngẫu nhiên một bạn học (partner) khác mình để trao đổi kiến thức
            partner_idx = np.array([np.random.choice(np.delete(np.arange(self.pop_size), i)) 
                                    for i in range(self.pop_size)])
            
            # Nếu bạn học giỏi hơn -> học theo hướng của bạn; ngược lại -> học tránh hướng của bạn
            learn_dir = np.where((fitness < fitness[partner_idx])[:, np.newaxis], 
                                 pop - pop[partner_idx], 
                                 pop[partner_idx] - pop)
            
            new_pop_l = np.clip(pop + np.random.rand(self.pop_size, dim) * learn_dir, lb, ub)
            new_fit_l = np.apply_along_axis(self.problem.fitness, 1, new_pop_l)
            
            # Cập nhật nếu kết quả sau khi trao đổi nhóm tốt hơn
            better_mask_l = new_fit_l < fitness
            pop[better_mask_l] = new_pop_l[better_mask_l]
            fitness[better_mask_l] = new_fit_l[better_mask_l]

            # Cập nhật Best toàn cục sau mỗi vòng lặp
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.global_best_fitness:
                self.update_global_best(pop[best_idx], fitness[best_idx])
            
            # Lưu lịch sử kèm quần thể để tính Diversity (Exploration vs Exploitation)
            self.save_history(pop)

        return self.global_best_solution, self.global_best_fitness