import numpy as np
from algorithms.optimizer import Optimizer

class HarmonySearch(Optimizer):
    """
    Harmony Search (HS)
    Cảm hứng: Quá trình sáng tác nhạc (Jazz improvisation).
    Các tham số quan trọng:
    - HMCR (Harmony Memory Considering Rate): Xác suất chọn giá trị từ bộ nhớ.
    - PAR (Pitch Adjusting Rate): Xác suất điều chỉnh nhẹ (pitch adjustment) giá trị đã chọn.
    - BW (Bandwidth): Độ lớn bước điều chỉnh.
    """
    def __init__(self, problem, pop_size=20, hmcr=0.9, par=0.3, bw=0.01, **kwargs):
        # pop_size ở đây đóng vai trò là HMS (Harmony Memory Size)
        super().__init__(problem, pop_size=pop_size, **kwargs)
        self.hmcr = hmcr
        self.par = par
        self.bw = bw

    def _evolve(self):
        dim = self.problem.dim
        lb = self.problem.bounds[:, 0]
        ub = self.problem.bounds[:, 1]

        # 1. Khởi tạo Harmony Memory (HM)
        hm = np.random.uniform(lb, ub, (self.pop_size, dim))
        hm_fitness = np.apply_along_axis(self.problem.fitness, 1, hm)

        # Cập nhật Best ban đầu
        best_idx = np.argmin(hm_fitness)
        self.update_global_best(hm[best_idx], hm_fitness[best_idx])
        self.save_history()

        for _ in range(self.max_iter):
            # 2. Tạo một bản nhạc mới (New Harmony)
            new_harmony = np.zeros(dim)
            
            for i in range(dim):
                if np.random.rand() < self.hmcr:
                    # Memory Consideration: Chọn từ HM
                    rand_idx = np.random.randint(0, self.pop_size)
                    value = hm[rand_idx, i]
                    
                    # Pitch Adjustment: Điều chỉnh nhẹ
                    if np.random.rand() < self.par:
                        # Cộng hoặc trừ một lượng nhỏ bw
                        if np.random.rand() < 0.5:
                            value += np.random.rand() * self.bw
                        else:
                            value -= np.random.rand() * self.bw
                    
                    new_harmony[i] = value
                else:
                    # Random Selection: Chọn ngẫu nhiên trong miền giá trị
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])

            # Clip để đảm bảo nằm trong biên
            new_harmony = np.clip(new_harmony, lb, ub)
            new_fitness = self.problem.fitness(new_harmony)

            # 3. Cập nhật Harmony Memory
            # Tìm bản nhạc tệ nhất trong bộ nhớ
            worst_idx = np.argmax(hm_fitness)
            if new_fitness < hm_fitness[worst_idx]:
                hm[worst_idx] = new_harmony
                hm_fitness[worst_idx] = new_fitness

            # Cập nhật Global Best
            curr_best_idx = np.argmin(hm_fitness)
            if hm_fitness[curr_best_idx] < self.global_best_fitness:
                self.update_global_best(hm[curr_best_idx], hm_fitness[curr_best_idx])
            
            self.save_history()

        return self.global_best_solution, self.global_best_fitness