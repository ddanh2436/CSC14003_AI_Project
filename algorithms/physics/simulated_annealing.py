import numpy as np
from algorithms.optimizer import Optimizer

class SimulatedAnnealing(Optimizer):
    """
    Simulated Annealing (SA) - Thuật toán Tôi luyện thép
    Thuộc nhóm: Physics-based / Local Search
    
    Cơ chế: 
    - Khác với Hill Climbing (chỉ leo lên), SA đôi khi chấp nhận bước đi "xấu hơn" 
      để có cơ hội thoát khỏi cực trị địa phương (Local Optima).
    - Xác suất chấp nhận cái xấu phụ thuộc vào "Nhiệt độ" (Temperature).
    - Nhiệt độ cao (đầu game) -> Dễ dãi. Nhiệt độ thấp (cuối game) -> Khắt khe.
    """
    def __init__(self, problem, initial_temp=1000, cooling_rate=0.95, step_size=0.1, **kwargs):
        """
        Args:
            initial_temp: Nhiệt độ khởi tạo (Càng cao càng dễ chấp nhận lỗi ở đầu)
            cooling_rate: Tốc độ làm nguội (0.8 - 0.99). Thường dùng 0.95 hoặc 0.99
            step_size: Độ lớn bước nhảy khi tìm hàng xóm
        """
        super().__init__(problem, **kwargs)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.step_size = step_size

    def _evolve(self):
        # 1. Khởi tạo điểm bắt đầu ngẫu nhiên
        current_sol = np.random.uniform(
            self.problem.bounds[:, 0], 
            self.problem.bounds[:, 1]
        )
        current_fit = self.problem.fitness(current_sol)
        
        # Cập nhật Global Best ban đầu
        self.update_global_best(current_sol, current_fit)
        self.save_history() # Lưu lại fitness hiện tại vào lịch sử
        
        # Thiết lập nhiệt độ ban đầu
        temp = self.initial_temp

        # 2. Vòng lặp tối ưu (Quá trình làm nguội)
        for _ in range(self.max_iter):
            # --- TẠO ỨNG VIÊN (NEIGHBOR) ---
            # Cộng nhiễu Gaussian để tạo điểm lân cận
            neighbor = current_sol + np.random.normal(0, self.step_size, size=self.problem.dim)
            
            # Đảm bảo điểm mới vẫn nằm trong giới hạn bài toán
            neighbor = np.clip(neighbor, self.problem.bounds[:, 0], self.problem.bounds[:, 1])
            
            # Tính fitness điểm mới
            neighbor_fit = self.problem.fitness(neighbor)

            # --- QUYẾT ĐỊNH CHẤP NHẬN HAY KHÔNG? ---
            # Tính độ chênh lệch năng lượng (Delta E)
            delta = neighbor_fit - current_fit

            if delta < 0:
                # TRƯỜNG HỢP 1: Tốt hơn (Xuống dốc) -> LUÔN CHẤP NHẬN
                current_sol = neighbor
                current_fit = neighbor_fit
                
                # Cập nhật kết quả tốt nhất toàn cục nếu phá kỷ lục
                if current_fit < self.global_best_fitness:
                    self.update_global_best(current_sol, current_fit)
            else:
                # TRƯỜNG HỢP 2: Tệ hơn (Lên dốc) -> CHẤP NHẬN CÓ XÁC SUẤT
                # Công thức Metropolis: P = exp(-delta / T)
                # T càng lớn -> P càng gần 1 (Dễ chấp nhận)
                # T càng nhỏ -> P càng gần 0 (Khó chấp nhận)
                probability = np.exp(-delta / temp)
                
                # Tung đồng xu ngẫu nhiên
                if np.random.rand() < probability:
                    current_sol = neighbor
                    current_fit = neighbor_fit

            # --- LÀM NGUỘI ---
            # Giảm nhiệt độ theo hệ số cooling_rate
            temp *= self.cooling_rate
            
            # Lưu lịch sử để vẽ biểu đồ
            self.save_history()

        return self.global_best_solution, self.global_best_fitness