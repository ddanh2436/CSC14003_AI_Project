import numpy as np
from algorithms.optimizer import Optimizer

class HillClimbingTSP(Optimizer):
    """
    Hill Climbing chuyên dụng cho bài toán rời rạc (TSP).
    Thay vì cộng nhiễu Gaussian, ta dùng phép SWAP (Đổi chỗ 2 thành phố).
    """
    def __init__(self, problem, max_iter=1000, **kwargs):
        super().__init__(problem, max_iter=max_iter, **kwargs)

    def _evolve(self):
        # 1. Khởi tạo: Một hoán vị ngẫu nhiên các thành phố
        # Ví dụ: [0, 1, 2, ..., 19] -> [5, 2, 19, ..., 0]
        current_sol = np.random.permutation(self.problem.n_cities)
        current_fit = self.problem.fitness(current_sol)
        
        self.update_global_best(current_sol, current_fit)
        self.save_history()

        # 2. Vòng lặp tối ưu
        for _ in range(self.max_iter):
            # --- TẠO HÀNG XÓM (Swap Mutation) ---
            neighbor = current_sol.copy()
            
            # Chọn ngẫu nhiên 2 vị trí khác nhau để đổi chỗ
            idx1, idx2 = np.random.choice(self.problem.n_cities, 2, replace=False)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
            
            # Tính fitness
            neighbor_fit = self.problem.fitness(neighbor)

            # --- LEO ĐỒI (Chỉ chấp nhận nếu tốt hơn) ---
            if neighbor_fit < current_fit:
                current_sol = neighbor
                current_fit = neighbor_fit
                
                if current_fit < self.global_best_fitness:
                    self.update_global_best(current_sol, current_fit)
            
            self.save_history()
            
        return self.global_best_solution, self.global_best_fitness