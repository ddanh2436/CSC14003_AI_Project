import numpy as np
import time

class Optimizer:
    """
    Class cha (Base Class) cho tất cả các thuật toán tối ưu.
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
        self.pop_size = kwargs.get('pop_size', 30) # Dùng cho các thuật toán bầy đàn/tiến hóa
        
        # Lưu lịch sử fitness tốt nhất qua từng vòng lặp (để vẽ biểu đồ)
        self.history = []     
        self.run_time = 0     
        
        # Kết quả tốt nhất tìm được
        self.global_best_solution = None
        self.global_best_fitness = -np.inf if maximize else np.inf

    def solve(self):
        """
        Hàm khung sườn để chạy thuật toán.
        """
        start_time = time.time()
        
        # Gọi hàm xử lý chính (các class con sẽ phải tự định nghĩa hàm này)
        solution, fitness = self._evolve() 
        
        end_time = time.time()
        self.run_time = end_time - start_time
        
        # Trả về: Giải pháp tốt nhất, Fitness tốt nhất, Lịch sử hội tụ
        return solution, fitness, self.history

    def _evolve(self):
        """Logic riêng của từng thuật toán sẽ nằm ở đây (Abstract method)"""
        raise NotImplementedError("Lỗi: Bạn chưa viết hàm _evolve() cho thuật toán này!")

    def update_global_best(self, solution, fitness):
        """Hàm hỗ trợ cập nhật kết quả tốt nhất (Dùng chung cho mọi thuật toán)"""
        # Kiểm tra xem kết quả mới có tốt hơn kết quả cũ không
        if self.maximize:
            is_better = fitness > self.global_best_fitness
        else:
            is_better = fitness < self.global_best_fitness
        
        if is_better:
            self.global_best_fitness = fitness
            # .copy() là bắt buộc với NumPy để tránh lỗi tham chiếu bộ nhớ
            self.global_best_solution = solution.copy() 
            
    def save_history(self):
        """Lưu fitness tốt nhất hiện tại vào lịch sử"""
        self.history.append(self.global_best_fitness)

def calculate_diversity(self, population):
        """
        Tính độ đa dạng của quần thể (Dùng cho GA, PSO, DE...).
        Công thức: Trung bình khoảng cách từ các cá thể đến trọng tâm (center).
        """
        if population is None or len(population) == 0:
            return 0
            
        # Tính điểm trung tâm của quần thể
        center = np.mean(population, axis=0)
        
        # Tính khoảng cách Euclidean từ mỗi cá thể đến tâm
        distances = np.linalg.norm(population - center, axis=1)
        
        # Trả về khoảng cách trung bình
        return np.mean(distances)