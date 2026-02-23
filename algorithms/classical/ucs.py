import numpy as np
import heapq
from algorithms.optimizer import Optimizer

class UniformCostSearch(Optimizer):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)

    def _evolve(self):
        start = int(self.problem.start_node)
        goal = int(self.problem.goal_node)
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        visited = set()

        self.save_history()

        while open_set:
            current_g, current = heapq.heappop(open_set)
            current = int(current)
            
            # Nếu đã xử lý rồi thì bỏ qua (tránh lặp vô hạn)
            if current in visited:
                continue
            visited.add(current)
            
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                fitness = self.problem.fitness(path)
                self.update_global_best(np.array(path), fitness)
                self.save_history()
                return self.global_best_solution, self.global_best_fitness
                
            for neighbor in self.problem.get_neighbors(current):
                neighbor = int(neighbor)
                tentative_g_score = current_g + self.problem.get_cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (tentative_g_score, neighbor))
                    
        return None, float('inf')

    def _reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]