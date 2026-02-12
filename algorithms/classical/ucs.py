import numpy as np
import heapq
from algorithms.optimizer import Optimizer

class UniformCostSearch(Optimizer):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, maximize=False, **kwargs)
    def _evolve(self):
        start = self.problem.start_node
        goal = self.problem.goal_node
        
        open_set = [(0, start)]
        parent = {start: None}
        g_score = {start: 0}
        
        while open_set:
            current_g, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstructed_path(parent, goal)
            for neighbor in self.problem.get_neighbors(current):
                new_g = current_g + 1
                if neighbor not in g_score or new_g < g_score[neighbor]:
                    g_score[neighbor] = new_g
                    parent[neighbor] = current
                    heapq.heappush(open_set, (new_g, neighbor))
                self.save_history()
        return None, float('inf')
    def _reconstruct_path(self, parent, goal):
        path = []   
        curr = self.problem.goal
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path = path[::-1]
        self.update_global_best(np.array(path), len(path))
        return self.global_best_solution, self.global_best_fitness