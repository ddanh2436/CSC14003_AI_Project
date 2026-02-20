import numpy as np
from collections import deque
from algorithms.optimizer import Optimizer

class BreadthFirstSearch(Optimizer):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)

    def _evolve(self):
        # Ép kiểu chuẩn Python (int) ngay từ đầu để tránh lỗi NumPy
        start = int(self.problem.start_node)
        goal = int(self.problem.goal_node)

        queue = deque([start])
        came_from = {}
        visited = {start}

        self.save_history()

        while queue:
            current = int(queue.popleft())

            if current == goal:
                path = self._reconstruct_path(came_from, current)
                fitness = self.problem.fitness(path)
                
                self.update_global_best(np.array(path), fitness)
                self.save_history()
                return self.global_best_solution, self.global_best_fitness

            for neighbor in self.problem.get_neighbors(current):
                neighbor = int(neighbor) # Ép kiểu bảo vệ 
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)

        return None, float('inf')

    def _reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]