import numpy as np
from collections import deque
from algorithms.optimizer import Optimizer

class BreadthFirstSearch(Optimizer):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, maximize=False, **kwargs)
    
    def _evolve(self):
        start = self.problem.start_node
        goal = self.problem.goal_node 
        queue = deque([start])
        parent = {start: None}
        
        while queue: 
            current = queue.popleft()
            
            if current == goal:
                return self._finalize_result(parent, goal)
            for neighbor in self.problem.get_neighbors(current):
                if neighbor not in parent: 
                    parent[neighbor] = current
                    queue.append(neighbor)
            
            self.save_history()
        return None, float('inf')
    def _finalize_result(self, parent, goal): 
        path = []
        curr = goal
        while curr: 
            path.append(curr)
            curr = parent[curr]
        path = path[::-1]
        self.update_global_best(np.array(path), len(path))
        return self.global_best_solution, self.global_best_fitness