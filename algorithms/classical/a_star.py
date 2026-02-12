import numpy as np
import heapq
from algorithms.optimizer import Optimizer

class AStarSearch(Optimizer):
    def __init__(self, problem, **kwargs):
        # A* thường tìm đường đi ngắn nhất (Minimize)
        super().__init__(problem, maximize = False, **kwargs)
    def _evolve(self):
        """
        Triển khai thuật toán A* tìm đường đi ngắn nhất từ start đến goal.
        """
        start = self.problem.start_node
        goal = self.problem.goal_node
        
        # Priority Queue lưu: (f_score, node)
        # f(n) = g(n) + h(n)
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        # Save the route
        came_from = {}
        
        # The cost from start to current node 
        g_score = {start: 0}
        
        # Count the total cost
        f_score = {start: self.problem.heuristic(start)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Find the goal, recreate the path
                path = self._reconstruct_path(came_from, current)
                fitness = self.problem.fitness(path)
                self.update_global_best(np.array(path), fitness)
                self.save_history()
                return self.global_best_solution, self.global_best_fitness
            # Traversal all neighbors
            for neighbor in self.problem.get_neighbors(current):
                # The cost from start to neighbor through current node (each step = 1)
                tentative_g_score = g_score[current] + 1
                
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current 
                    g_score = tentative_g_score
                    f_score = tentative_g_score + self.problem.heuristic(neighbor)
                    
                    if neighbor not in [n[1] for n in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
            self.global_best_fitness = g_score.get(current, float('inff'))
            self.save_hisstory()
        return None, float('inf')
    def _reconstructed_path(self, came_from, current):
        path = [current]
        while current in came_from: 
            current = came_from[current]
            path.append(current)
        return path[::-1] # Reverse from goal -> start to have the right order
    