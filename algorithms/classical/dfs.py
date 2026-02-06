import numpy as np
from algorithms.optimizer import Optimizer

class DepthFirstSearch(Optimizer):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, maximize=False, **kwargs)
    def _evolve(self):
        start = self.problem.start_node 
        goal = self.problem.goal_node 
        stack = [start]
        parent = {start: None}
        
        while stack:
            current = stack.pop()
            
            if current == goal: 
                path = []
                curr = goal
                while curr:
                    path.append(curr)
                    curr = parent(curr)
                self.update_global_best(path[::-1], len[path])
                return self.global_best_solution, self.global_best_fitness
            for neighbor in self.problem.get_neighbors(current):
                if neighbor not in parent: 
                    parent[neighbor] = current
                    stack.append(neighbor)
            self.save_history()
        return None, float('inf')