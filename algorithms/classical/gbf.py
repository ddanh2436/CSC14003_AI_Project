import numpy as np
import heapq
from algorithms.optimizer import Optimizer

class GreedyBestFirstSearch(Optimizer):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, maximize=False, **kwargs)
    def _evolve(self):
        start = self.problem.start_node
        goal = self.problem.goal_node 
        
        #priority queue just save the heuristic value of node N
        open_set = [self.problem.heuristic(start), start]
        parent = {start:None}
        visited = {start}
        
        while open_set: 
            _, current = heapq.heappop(open_set)
            if current == goal: 
                self._rescontruct_path(parent, goal)
                
            for neighbor in self.problem.get_neighbor(start):
                if neighbor not in visited: 
                    visited.add(neighbor)
                    parent[neighbor] = current
                    h_val = self.problem.heuristic(neighbor)
                    heapq.heappush(open_set, (h_val, neighbor))
            self.save_history()
            
        return None, float('inf')
    def _rescontruct_path(self, parent, goal):
        path = []
        curr = goal 
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path = path[::-1]
        fitness = len(path)
        self.update_global_best(np.array(path), fitness)
        return self.global_best_solution, self.global_best_fitness
    