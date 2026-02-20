import numpy as np
import heapq
from algorithms.optimizer import Optimizer

class AStarSearch(Optimizer):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)

    def _evolve(self):
        # Đảm bảo start và goal là số nguyên chuẩn của Python
        start = int(self.problem.start_node)
        goal = int(self.problem.goal_node)

        # Hàng đợi ưu tiên lưu tuple (f_score, node)
        open_set = []
        heapq.heappush(open_set, (self.problem.heuristic(start), start))

        # Khởi tạo DICTIONARY bằng ngoặc nhọn {}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.problem.heuristic(start)}
        open_set_hash = {start}

        self.save_history()

        while len(open_set) > 0:
            # Lấy node có f_score nhỏ nhất
            current = int(heapq.heappop(open_set)[1])
            open_set_hash.remove(current)

            # Nếu đã đến đích
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                fitness = self.problem.fitness(path)
                
                # Cập nhật kết quả tốt nhất
                self.update_global_best(np.array(path), fitness)
                self.save_history()
                return self.global_best_solution, self.global_best_fitness

            # Duyệt qua các hàng xóm
            for neighbor in self.problem.get_neighbors(current):
                # Ép kiểu neighbor về số nguyên (Tránh lỗi numpy scalar)
                neighbor = int(neighbor)

                # Chi phí đi qua current để đến neighbor
                tentative_g_score = g_score[current] + self.problem.get_cost(current, neighbor)

                # Nếu tìm được đường đi ngắn hơn đến neighbor
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.problem.heuristic(neighbor)

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        # Nếu không tìm thấy đường đi
        return None, float('inf')

    def _reconstruct_path(self, came_from, current):
        """Hàm truy xuất ngược lại đường đi từ đích về bắt đầu"""
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1] # Đảo ngược list để có đường đi từ Start -> Goal