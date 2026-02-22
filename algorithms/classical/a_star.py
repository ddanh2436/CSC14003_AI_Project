import numpy as np
import heapq
from algorithms.optimizer import Optimizer

class AStarSearch(Optimizer):
    def __init__(self, problem, visualize=False, pause_time=0.05, **kwargs):
        super().__init__(problem, **kwargs)
        self.visualize = visualize
        self.pause_time = pause_time

    def _evolve(self):
        start = int(self.problem.start_node)
        goal = int(self.problem.goal_node)

        open_set = []
        heapq.heappush(open_set, (self.problem.heuristic(start), start))

        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.problem.heuristic(start)}
        open_set_hash = {start}

        self.save_history()

        while len(open_set) > 0:
            current = int(heapq.heappop(open_set)[1])
            open_set_hash.remove(current)

            # --- THÊM PHẦN ANIMATION TẠI ĐÂY ---
            if self.visualize:
                self.problem.visualize_step(
                    visited=set(came_from.keys()), 
                    frontier=open_set_hash, 
                    current=current, 
                    pause_time=self.pause_time
                )

            if current == goal:
                path = self._reconstruct_path(came_from, current)
                fitness = self.problem.fitness(path)
                
                # --- VẼ KẾT QUẢ ĐƯỜNG ĐI CUỐI CÙNG ---
                if self.visualize:
                    self.problem.visualize_step(
                        visited=set(came_from.keys()), 
                        frontier=open_set_hash, 
                        current=current, 
                        path=path,
                        pause_time=self.pause_time
                    )
                
                self.update_global_best(np.array(path), fitness)
                self.save_history()
                return self.global_best_solution, self.global_best_fitness

            for neighbor in self.problem.get_neighbors(current):
                neighbor = int(neighbor)
                tentative_g_score = g_score[current] + self.problem.get_cost(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.problem.heuristic(neighbor)

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        return None, float('inf')

    # (Giữ nguyên hàm _reconstruct_path của bạn)
    def _reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]