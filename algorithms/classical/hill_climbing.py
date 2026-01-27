import numpy as np
# Import class cha để kế thừa
from algorithms.optimizer import Optimizer

class HillClimbing(Optimizer):
    def __init__(self, problem, step_size=0.1, max_iter=1000, **kwargs):
        super().__init__(problem, **kwargs)
        self.step_size = step_size # Độ lớn bước nhảy

    def _evolve(self):
        # 1. Khởi tạo ngẫu nhiên một điểm bắt đầu
        # np.random.uniform(low, high) tạo số thực ngẫu nhiên
        current_solution = np.random.uniform(
            self.problem.bounds[:, 0], 
            self.problem.bounds[:, 1]
        )
        current_fitness = self.problem.fitness(current_solution)

        # Cập nhật Global Best ban đầu
        self.update_global_best(current_solution, current_fitness)
        self.save_history()

        # 2. Vòng lặp tối ưu
        for _ in range(self.max_iter):
            # Tạo ứng viên mới bằng cách cộng nhiễu (Gaussian noise) vào vị trí hiện tại
            candidate = current_solution + np.random.normal(0, self.step_size, size=self.problem.dim)
            
            # Đảm bảo ứng viên vẫn nằm trong giới hạn bài toán (Clip)
            candidate = np.clip(candidate, self.problem.bounds[:, 0], self.problem.bounds[:, 1])
            
            candidate_fitness = self.problem.fitness(candidate)

            # --- LOGIC LEO ĐỒI ---
            # Nếu ứng viên mới tốt hơn hiện tại -> Di chuyển tới đó
            if candidate_fitness < current_fitness: # Giả sử bài toán tìm Min
                current_solution = candidate
                current_fitness = candidate_fitness
                
                # Cập nhật kết quả tốt nhất toàn cục
                self.update_global_best(current_solution, current_fitness)
            
            # Lưu lịch sử (để vẽ biểu đồ)
            self.save_history()
        
        return self.global_best_solution, self.global_best_fitness