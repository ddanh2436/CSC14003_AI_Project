import numpy as np
import heapq
from algorithms.optimizer import Optimizer

class GreedyBestFirstSearch(Optimizer):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)

    def _evolve(self):
        start = int(self.problem.start_node)
        goal = int(self.problem.goal_node)
        
        open_set = []
        # GBFS chỉ dùng h(n), KHÔNG CỘNG g(n) như A*
        heapq.heappush(open_set, (self.problem.heuristic(start), start))
        came_from = {}
        visited = {start}

        self.save_history()
        
        while open_set: 
            _, current = heapq.heappop(open_set)
            current = int(current)
            
            if current == goal: 
                path = self._reconstruct_path(came_from, current)
                fitness = self.problem.fitness(path)
                self.update_global_best(np.array(path), fitness)
                self.save_history()
                return self.global_best_solution, self.global_best_fitness
                
            for neighbor in self.problem.get_neighbors(current):
                neighbor = int(neighbor)
                if neighbor not in visited: 
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    h_val = self.problem.heuristic(neighbor)
                    heapq.heappush(open_set, (h_val, neighbor))
                    
        return None, float('inf')

    def _reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]