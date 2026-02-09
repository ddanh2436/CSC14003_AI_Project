import numpy as np
import time

class Optimizer:
    """
    Class cha (Base Class) cho tất cả các thuật toán tối ưu.
    Cung cấp các khung sườn cơ bản về quản lý lịch sử, thời gian và kết quả.
    """
    def __init__(self, problem, maximize=False, **kwargs):
        """
        Args:
            problem: Object chứa thông tin bài toán (hàm mục tiêu, giới hạn...)
            maximize: True nếu tìm Max, False nếu tìm Min (Mặc định là Min)
            kwargs: Các tham số khác (pop_size, max_iter...)
        """
        self.problem = problem
        self.maximize = maximize
        
        # Lấy tham số cấu hình, nếu không có thì dùng mặc định
        self.max_iter = kwargs.get('max_iter', 100)
        self.pop_size = kwargs.get('pop_size', 30)
        
        # Lưu lịch sử fitness tốt nhất qua từng vòng lặp (để vẽ biểu đồ hội tụ)
        self.history = []     
        # Lưu lịch sử độ đa dạng (để phân tích Exploration vs Exploitation)
        self.diversity_history = []
        self.run_time = 0     
        
        # Kết quả tốt nhất tìm được
        self.global_best_solution = None
        self.global_best_fitness = -np.inf if maximize else np.inf

    def solve(self):
        """
        Hàm khung sườn để thực thi thuật toán.
        """
        start_time = time.time()
        
        # Gọi hàm xử lý chính (các class con sẽ định nghĩa hàm này)
        solution, fitness = self._evolve() 
        
        end_time = time.time()
        self.run_time = end_time - start_time
        
        return solution, fitness, self.history

    def _evolve(self):
        """Logic riêng của từng thuật toán (Abstract method)"""
        raise NotImplementedError("Lỗi: Bạn chưa viết hàm _evolve() cho thuật toán này!")

    def update_global_best(self, solution, fitness):
        """Cập nhật kết quả tốt nhất toàn cục nếu tìm thấy giải pháp tốt hơn"""
        if self.maximize:
            is_better = fitness > self.global_best_fitness
        else:
            is_better = fitness < self.global_best_fitness
        
        if is_better:
            self.global_best_fitness = fitness
            # .copy() là bắt buộc với NumPy để tránh lỗi tham chiếu bộ nhớ
            self.global_best_solution = solution.copy() 
            
    def save_history(self, population=None):
        """
        Lưu fitness tốt nhất hiện tại và tính toán độ đa dạng của quần thể.
        """
        self.history.append(self.global_best_fitness)
        
        # Nếu có thông tin quần thể, tính độ đa dạng (Diversity index)
        if population is not None and len(population) > 0:
            diversity = self.calculate_diversity(population)
            self.diversity_history.append(diversity)

    def calculate_diversity(self, population):
        """
        Tính độ đa dạng của quần thể dựa trên khoảng cách Euclidean trung bình đến trọng tâm.
        Giúp minh chứng hành vi Exploration (đa dạng cao) vs Exploitation (đa dạng thấp).
        """
        if population is None or len(population) == 0:
            return 0
        
        # Tính điểm trọng tâm của quần thể
        center = np.mean(population, axis=0)
        # Tính khoảng cách từ mỗi cá thể đến tâm
        distances = np.linalg.norm(population - center, axis=1)
        # Trả về khoảng cách trung bình
        return np.mean(distances)