import numpy as np
from collections import deque
from algorithms.optimizer import Optimizer

class BreadthFirstSearch(Optimizer):
    def __init__(self, problem, visualize=False, pause_time=0.05, **kwargs):
        super().__init__(problem, **kwargs)
        self.visualize = visualize
        self.pause_time = pause_time

    def _evolve(self):
        start = int(self.problem.start_node)
        goal = int(self.problem.goal_node)

        queue = deque([start])
        came_from = {}
        visited = {start}

        self.save_history()

        while queue:
            current = int(queue.popleft())

            # --- GỌI HIỂN THỊ ANIMATION ---
            if self.visualize:
                self.problem.visualize_step(
                    visited=visited, 
                    frontier=set(queue), 
                    current=current, 
                    pause_time=self.pause_time
                )

            if current == goal:
                path = self._reconstruct_path(came_from, current)
                fitness = self.problem.fitness(path)
                
                # --- VẼ ĐƯỜNG ĐI CUỐI CÙNG ---
                if self.visualize:
                    self.problem.visualize_step(
                        visited=visited, 
                        frontier=set(queue), 
                        current=current, 
                        path=path,
                        pause_time=self.pause_time
                    )
                
                self.update_global_best(np.array(path), fitness)
                self.save_history()
                return self.global_best_solution, self.global_best_fitness

            for neighbor in self.problem.get_neighbors(current):
                neighbor = int(neighbor)
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