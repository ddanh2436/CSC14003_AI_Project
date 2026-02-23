import numpy as np
import heapq
import matplotlib.pyplot as plt
from algorithms.optimizer import Optimizer

class AStarSearch(Optimizer):
    def __init__(self, problem, visualize=False, pause_time=0.03, **kwargs):
        super().__init__(problem, **kwargs)
        self.visualize = visualize
        self.pause_time = pause_time

    def _init_plot(self):
        if not self.visualize: return
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 8), num="A* Search Running...")
        
        self.ax.set_title(f"A* Search Animation ({self.problem.n_nodes} nodes)", fontsize=14, fontweight='bold')
        
        # Vẽ các cạnh mờ nền
        for i in range(self.problem.n_nodes):
            for j in range(i + 1, self.problem.n_nodes):
                if self.problem.adj_matrix[i][j] > 0:
                    self.ax.plot([self.problem.coords[i][0], self.problem.coords[j][0]], 
                                 [self.problem.coords[i][1], self.problem.coords[j][1]], 
                                 color='lightgray', alpha=0.3, zorder=1)
        
        # Khởi tạo các điểm scatter
        self.visited_scatter = self.ax.scatter([], [], c='lightblue', s=60, label='Visited (Đã xét)', zorder=3)
        self.frontier_scatter = self.ax.scatter([], [], c='orange', s=60, label='Frontier (Tập mở)', zorder=4)
        self.current_scatter = self.ax.scatter([], [], c='yellow', edgecolors='red', linewidths=2, s=150, label='Current (Đang xét)', zorder=5)
        
        # Điểm bắt đầu và kết thúc
        start, goal = int(self.problem.start_node), int(self.problem.goal_node)
        self.ax.scatter(self.problem.coords[start][0], self.problem.coords[start][1], c='green', marker='s', s=150, label='Start', zorder=6)
        self.ax.scatter(self.problem.coords[goal][0], self.problem.coords[goal][1], c='red', marker='*', s=250, label='Goal', zorder=6)
        
        # Nhãn thuật toán to đùng
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        self.ax.text(0.02, 0.98, "ALGORITHM: A* SEARCH", transform=self.ax.transAxes,
                     fontsize=16, fontweight='bold', verticalalignment='top', bbox=props, color='darkblue', zorder=20)
        
        self.ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.tight_layout()
        plt.draw()

    def _update_plot(self, current, visited, frontier):
        if not self.visualize: return
        
        if visited: self.visited_scatter.set_offsets(self.problem.coords[list(visited)])
        if frontier: self.frontier_scatter.set_offsets(self.problem.coords[list(frontier)])
        self.current_scatter.set_offsets(self.problem.coords[current])
        
        self.fig.canvas.flush_events()
        plt.pause(self.pause_time)

    def _evolve(self):
        # Ép kiểu int để chống lỗi với Numpy
        start, goal = int(self.problem.start_node), int(self.problem.goal_node)

        open_set = []
        heapq.heappush(open_set, (self.problem.heuristic(start), start))

        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.problem.heuristic(start)}
        
        # Dùng set để hỗ trợ animation (biết đỉnh nào đang ở biên, đỉnh nào đã duyệt)
        open_set_hash = {start}
        closed_set = set() 

        self._init_plot()
        self.save_history()

        while len(open_set) > 0:
            current = int(heapq.heappop(open_set)[1])
            
            # Xử lý tập hợp cho Animation
            if current in open_set_hash:
                open_set_hash.remove(current)
            closed_set.add(current)

            self._update_plot(current, closed_set, open_set_hash)

            if current == goal:
                path = self._reconstruct_path(came_from, current)
                fitness = self.problem.fitness(path)
                
                self.update_global_best(np.array(path), fitness)
                self.save_history()
                
                # Vẽ đường line chốt hạ
                if self.visualize:
                    route = self.problem.coords[path]
                    self.ax.plot(route[:, 0], route[:, 1], c='blue', linewidth=4, label='Path', zorder=10)
                    self.ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
                    plt.ioff()
                    plt.show()
                return self.global_best_solution, self.global_best_fitness

            for neighbor in self.problem.get_neighbors(current):
                neighbor = int(neighbor) # Ép kiểu int để chống lỗi Numpy
                tentative_g_score = g_score[current] + self.problem.get_cost(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.problem.heuristic(neighbor)

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        if self.visualize:
            plt.ioff()
            plt.show()
        return None, float('inf')

    def _reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]