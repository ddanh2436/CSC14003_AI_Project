import numpy as np
from algorithms.optimizer import Optimizer

class GravitationalSearchAlgorithm(Optimizer):
    """
    Gravitational Search Algorithm (GSA)
    Cảm hứng: Định luật vạn vật hấp dẫn Newton.
    Cơ chế: Các vật thể hút nhau, vật có khối lượng lớn (fitness tốt) sẽ di chuyển chậm và hút các vật khác về phía mình.
    """
    def __init__(self, problem, pop_size=50, G0=100, alpha=20, **kwargs):
        """
        G0: Hằng số hấp dẫn ban đầu
        alpha: Hằng số suy giảm (Decay constant)
        """
        super().__init__(problem, pop_size=pop_size, **kwargs)
        self.G0 = G0
        self.alpha = alpha

    def _evolve(self):
        dim = self.problem.dim
        # 1. Khởi tạo vị trí và vận tốc
        X = np.random.uniform(self.problem.bounds[:, 0], self.problem.bounds[:, 1], (self.pop_size, dim))
        V = np.zeros((self.pop_size, dim))
        fitness = np.apply_along_axis(self.problem.fitness, 1, X)

        # Cập nhật Best ban đầu
        best_idx = np.argmin(fitness)
        self.update_global_best(X[best_idx], fitness[best_idx])
        self.save_history()

        for t in range(self.max_iter):
            # 2. Cập nhật Hằng số hấp dẫn G(t) giảm dần theo thời gian
            G = self.G0 * np.exp(-self.alpha * t / self.max_iter)

            # 3. Tính khối lượng (Mass) của từng vật thể
            # Công thức: M_i = (fit_i - worst) / (best - worst)
            best_val = np.min(fitness)
            worst_val = np.max(fitness)
            
            # Tránh chia cho 0
            if worst_val == best_val:
                M = np.ones(self.pop_size)
            else:
                M = (fitness - worst_val) / (best_val - worst_val) # Với bài toán Min: (worst - fit) / (worst - best) ?
                # Sửa lại cho bài toán Minimization:
                # fit càng nhỏ (tốt) -> Mass càng to
                M = (worst_val - fitness) / (worst_val - best_val)

            # Chuẩn hóa Mass
            M = M / (np.sum(M) + 1e-10)

            # 4. Tính Gia tốc (Acceleration) a = F/M
            # F_ij = G * (M_i * M_j) / R * (x_j - x_i)
            # a_i = sum(F_ij) / M_i = sum( G * M_j / R * (x_j - x_i) )
            
            A = np.zeros((self.pop_size, dim))
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if i != j:
                        dist = np.linalg.norm(X[i] - X[j])
                        force = G * M[j] * (X[j] - X[i]) / (dist + 1e-10)
                        # Thêm một chút ngẫu nhiên vào lực hút (theo một số biến thể GSA)
                        rand_factor = np.random.rand(dim)
                        A[i] += rand_factor * force
            
            # 5. Cập nhật Vận tốc và Vị trí
            # V(t+1) = rand * V(t) + A(t)
            V = np.random.rand(self.pop_size, dim) * V + A
            X = X + V
            
            # Giới hạn không gian tìm kiếm
            X = np.clip(X, self.problem.bounds[:, 0], self.problem.bounds[:, 1])

            # 6. Đánh giá lại
            fitness = np.apply_along_axis(self.problem.fitness, 1, X)

            # Cập nhật Global Best
            curr_best_idx = np.argmin(fitness)
            if fitness[curr_best_idx] < self.global_best_fitness:
                self.update_global_best(X[curr_best_idx], fitness[curr_best_idx])
            
            self.save_history()

        return self.global_best_solution, self.global_best_fitness