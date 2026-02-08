import numpy as np
from algorithms.optimizer import Optimizer

class GravitationalSearchAlgorithm(Optimizer):
    """
    Gravitational Search Algorithm (GSA) - Phiên bản tối ưu hóa tốc độ.
    Cảm hứng: Định luật vạn vật hấp dẫn của Newton.
    Cơ chế: Các vật thể tương tác thông qua lực hấp dẫn; vật có khối lượng lớn (fitness tốt) 
    sẽ tạo lực hút mạnh hơn để kéo các vật thể khác về phía vùng không gian tốt hơn.
    """
    def __init__(self, problem, pop_size=50, G0=100, alpha=20, **kwargs):
        """
        Args:
            G0: Hằng số hấp dẫn ban đầu.
            alpha: Hằng số suy giảm hằng số hấp dẫn theo thời gian.
        """
        super().__init__(problem, pop_size=pop_size, **kwargs)
        self.G0 = G0
        self.alpha = alpha

    def _evolve(self):
        dim = self.problem.dim
        lb = self.problem.bounds[:, 0]
        ub = self.problem.bounds[:, 1]
        
        # 1. Khởi tạo vị trí và vận tốc
        X = np.random.uniform(lb, ub, (self.pop_size, dim))
        V = np.zeros((self.pop_size, dim))
        fitness = np.apply_along_axis(self.problem.fitness, 1, X)

        # Cập nhật Best ban đầu
        best_idx = np.argmin(fitness)
        self.update_global_best(X[best_idx], fitness[best_idx])
        self.save_history()

        for t in range(self.max_iter):
            # 2. Cập nhật Hằng số hấp dẫn G(t) giảm dần
            G = self.G0 * np.exp(-self.alpha * t / self.max_iter)

            # 3. Tính khối lượng (Mass) dựa trên Fitness
            best_val = np.min(fitness)
            worst_val = np.max(fitness)
            
            if worst_val == best_val:
                M = np.ones(self.pop_size)
            else:
                # Bài toán tối thiểu hóa: Fitness càng nhỏ -> Mass càng lớn
                M = (worst_val - fitness) / (worst_val - best_val + 1e-10)

            # Chuẩn hóa Mass để tổng bằng 1
            M = M / (np.sum(M) + 1e-10)

            # 4. TÍNH GIA TỐC (ACCELEARTION) - PHẦN ĐÃ VECTOR HÓA
            # Tính ma trận hiệu vị trí (X_j - X_i) cho mọi cặp i, j
            # Shape: (pop_size, pop_size, dim)
            diff = X[np.newaxis, :, :] - X[:, np.newaxis, :]
            
            # Tính ma trận khoảng cách Euclidean R_ij
            # Shape: (pop_size, pop_size)
            dist = np.linalg.norm(diff, axis=2) + 1e-10
            
            # Tính lực hấp dẫn F_ij và tổng hợp thành Gia tốc A_i
            # Công thức: a_i = sum_j (rand * G * M_j * (X_j - X_i) / R_ij)
            # Thêm ngẫu nhiên rand_factor cho từng chiều của lực hút
            rand_matrix = np.random.rand(self.pop_size, self.pop_size, dim)
            
            # Tính toán trọng số lực: G * M_j / R_ij
            # Shape: (pop_size, pop_size)
            force_weight = G * M[np.newaxis, :] / dist
            
            # Tổng hợp lực lên từng vật thể i theo trục axis=1 (tổng theo j)
            # A shape: (pop_size, dim)
            A = np.sum(rand_matrix * force_weight[:, :, np.newaxis] * diff, axis=1)

            # 5. Cập nhật Vận tốc và Vị trí
            V = np.random.rand(self.pop_size, dim) * V + A
            X = X + V
            
            # Giới hạn không gian tìm kiếm (Clipping)
            X = np.clip(X, lb, ub)

            # 6. Đánh giá lại Fitness và cập nhật Global Best
            fitness = np.apply_along_axis(self.problem.fitness, 1, X)
            curr_best_idx = np.argmin(fitness)
            
            if fitness[curr_best_idx] < self.global_best_fitness:
                self.update_global_best(X[curr_best_idx], fitness[curr_best_idx])
            
            self.save_history()

        return self.global_best_solution, self.global_best_fitness