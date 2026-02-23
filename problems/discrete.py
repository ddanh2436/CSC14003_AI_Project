import numpy as np
import matplotlib.pyplot as plt

class DiscreteProblem:
    """Class cha cho các bài toán rời rạc"""
    def __init__(self, name="Discrete Problem"):
        self.name = name
        self.bounds = None 
        self.dim = 0

    def fitness(self, solution):
        raise NotImplementedError

class TSP(DiscreteProblem):
    def __init__(self, n_cities=20, seed=42):
        super().__init__(name=f"TSP ({n_cities} cities)")
        self.n_cities = n_cities
        self.dim = n_cities
        
        np.random.seed(seed)
        self.cities = np.random.rand(n_cities, 2) * 100
        self.dist_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                self.dist_matrix[i, j] = np.linalg.norm(self.cities[i] - self.cities[j])

    def fitness(self, path):
        path = np.array(path, dtype=int)
        total_dist = 0
        for i in range(len(path) - 1):
            total_dist += self.dist_matrix[path[i], path[i+1]]
        total_dist += self.dist_matrix[path[-1], path[0]]
        return total_dist

    def visualize(self, path, title="TSP Route"):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=50, zorder=2, label='Cities')
        for i, (x, y) in enumerate(self.cities):
            plt.text(x + 1, y + 1, str(i), fontsize=9)
            
        path = np.array(path, dtype=int)
        closed_path = np.append(path, path[0])
        route_coords = self.cities[closed_path]
        plt.plot(route_coords[:, 0], route_coords[:, 1], c='blue', linestyle='-', linewidth=1, zorder=1, alpha=0.7)
        
        plt.title(f"{title}\nTotal Distance: {self.fitness(path):.2f}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

class Knapsack(DiscreteProblem):
    def __init__(self, n_items=50, capacity=None, seed=42):
        super().__init__(name=f"Knapsack ({n_items} items)")
        self.n_items = n_items
        self.dim = n_items
        self.bounds = np.array([[0.0, 1.0]] * n_items)
        
        np.random.seed(seed)
        self.weights = np.random.randint(1, 15, n_items)
        self.values = np.random.randint(10, 100, n_items)
        
        if capacity is None:
            self.capacity = int(np.sum(self.weights) * 0.5)
        else:
            self.capacity = capacity

    def fitness(self, solution):
        selected = np.round(solution).astype(int)
        total_weight = np.sum(selected * self.weights)
        
        if total_weight > self.capacity:
            ratios = self.values / self.weights
            selected_indices = np.where(selected == 1)[0]
            sorted_by_ratio = sorted(selected_indices, key=lambda idx: ratios[idx])
            
            for idx in sorted_by_ratio:
                selected[idx] = 0
                total_weight -= self.weights[idx]
                if total_weight <= self.capacity:
                    break
                    
        total_value = np.sum(selected * self.values)
        return -total_value

class ShortestPath(DiscreteProblem):
    def __init__(self, n_nodes=50, edge_prob=0.3, seed=42):
        super().__init__(name=f"Shortest Path ({n_nodes} nodes)")
        self.n_nodes = n_nodes
        self.start_node = 0
        self.goal_node = n_nodes - 1
        
        np.random.seed(seed)
        self.coords = np.random.rand(n_nodes, 2) * 100
        self.coords[self.start_node] = [0, 0]
        self.coords[self.goal_node] = [100, 100]

        self.adj_matrix = np.zeros((n_nodes, n_nodes))
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if np.random.rand() < edge_prob:
                    dist = np.linalg.norm(self.coords[i] - self.coords[j])
                    self.adj_matrix[i][j] = dist
                    self.adj_matrix[j][i] = dist
                    
        # ======================================================
        # FIX LỖI "ĐƯỜNG BAY THẲNG" (Bổ sung 2 dòng này)
        # Ép các thuật toán KHÔNG ĐƯỢC đi thẳng từ Start tới Goal
        # ======================================================
        self.adj_matrix[self.start_node][self.goal_node] = 0
        self.adj_matrix[self.goal_node][self.start_node] = 0

    def get_neighbors(self, node):
        return np.where(self.adj_matrix[node] > 0)[0]

    def get_cost(self, u, v):
        return self.adj_matrix[u][v]

    def heuristic(self, node):
        return np.linalg.norm(self.coords[node] - self.coords[self.goal_node])

    def fitness(self, path):
        if len(path) == 0 or path[0] != self.start_node or path[-1] != self.goal_node:
            return float('inf')
            
        cost = 0
        for i in range(len(path) - 1):
            u, v = int(path[i]), int(path[i+1])
            w = self.adj_matrix[u][v]
            if w == 0: 
                return float('inf') 
            cost += w
        return cost
    def visualize_paths(self, paths_dict, title="Shortest Path Comparison (All Algorithms)"):
        """
        Vẽ so sánh đường đi của nhiều thuật toán trên cùng một đồ thị.
        Dùng các nét vẽ khác nhau để chống trùng lặp (Overlap).
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))

        # 1. Vẽ toàn bộ các cạnh của đồ thị (Mờ và mỏng)
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if self.adj_matrix[i][j] > 0:
                    plt.plot([self.coords[i][0], self.coords[j][0]], 
                             [self.coords[i][1], self.coords[j][1]], 
                             color='lightgray', linewidth=0.5, alpha=0.4, zorder=1)

        # 2. Vẽ các đỉnh (Nodes)
        plt.scatter(self.coords[:, 0], self.coords[:, 1], c='white', edgecolors='gray', s=40, zorder=2)

        # Đánh dấu điểm Bắt đầu (Vuông Xanh) và Kết thúc (Sao Đỏ)
        plt.scatter(self.coords[self.start_node][0], self.coords[self.start_node][1],
                    c='green', marker='s', s=150, zorder=10, label='Start')
        plt.scatter(self.coords[self.goal_node][0], self.coords[self.goal_node][1],
                    c='red', marker='*', s=250, zorder=10, label='Goal')

        # 3. Cấu hình nét vẽ để chống trùng lặp (Overlap)
        # Các thuật toán sau sẽ có nét mỏng hơn và khác kiểu để nổi bật trên nền thuật toán trước
        colors = ['#d62728', '#9467bd', '#2ca02c', '#ff7f0e', '#1f77b4'] # Đỏ, Tím, Xanh lá, Cam, Xanh biển
        line_styles = ['-', '-', '--', '-.', ':'] 
        line_widths = [6, 4.5, 3, 2, 1.5] # Độ dày giảm dần
        alphas = [0.3, 0.5, 0.8, 1.0, 1.0]

        # 4. Vẽ đường đi của từng thuật toán
        for idx, (algo_name, path) in enumerate(paths_dict.items()):
            if path is None or len(path) == 0:
                continue

            path = np.array(path, dtype=int)
            route_coords = self.coords[path]

            c = colors[idx % len(colors)]
            ls = line_styles[idx % len(line_styles)]
            lw = line_widths[idx % len(line_widths)]
            alpha = alphas[idx % len(alphas)]

            plt.plot(route_coords[:, 0], route_coords[:, 1],
                     color=c, linestyle=ls, linewidth=lw, alpha=alpha,
                     zorder=3 + idx, 
                     label=f"{algo_name} (Cost: {self.fitness(path):.1f})")

        plt.title(title, fontsize=14, fontweight='bold')
        # Đưa chú thích ra góc ngoài cho đỡ che đồ thị
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def visualize_step(self, visited, frontier, current=None, path=None, pause_time=0.05):
        """
        Trực quan hóa từng bước của quá trình duyệt đồ thị.
        Dùng set_offsets để tối ưu FPS, không bị giật lag.
        """
        if not hasattr(self, 'fig'):
            plt.ion() # Bật chế độ vẽ tương tác
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            self.ax.set_title(self.name + " - Graph Search Animation")
            
            # 1. Vẽ TẤT CẢ các cạnh 1 lần duy nhất (Nền)
            for i in range(self.n_nodes):
                for j in range(i + 1, self.n_nodes):
                    if self.adj_matrix[i][j] > 0:
                        self.ax.plot([self.coords[i, 0], self.coords[j, 0]], 
                                     [self.coords[i, 1], self.coords[j, 1]], 
                                     color='lightgray', zorder=1, alpha=0.3)
            
            # 2. Vẽ tất cả các đỉnh
            self.ax.scatter(self.coords[:, 0], self.coords[:, 1], c='white', edgecolors='gray', zorder=2)
            
            # 3. Đánh dấu điểm Start và Goal
            self.ax.scatter(*self.coords[self.start_node], c='green', s=150, marker='s', label='Start', zorder=5)
            self.ax.scatter(*self.coords[self.goal_node], c='red', s=150, marker='*', label='Goal', zorder=5)
            
            # 4. Khởi tạo các đối tượng đồ họa trống
            self.visited_scatter = self.ax.scatter([], [], c='lightblue', s=60, zorder=3, label='Visited (Đã xét)')
            self.frontier_scatter = self.ax.scatter([], [], c='orange', s=60, zorder=4, label='Frontier (Tập mở)')
            self.current_scatter = self.ax.scatter([], [], c='yellow', s=120, edgecolors='red', linewidth=2, zorder=6, label='Current (Đang xét)')
            self.path_line, = self.ax.plot([], [], c='blue', linewidth=3, zorder=7, label='Path')
            
            self.ax.legend(loc='upper right')

        # --- CẬP NHẬT DỮ LIỆU ĐỘNG ---
        v_coords = self.coords[list(visited)] if visited else np.empty((0, 2))
        f_coords = self.coords[list(frontier)] if frontier else np.empty((0, 2))
        
        self.visited_scatter.set_offsets(v_coords)
        self.frontier_scatter.set_offsets(f_coords)
        
        if current is not None:
            self.current_scatter.set_offsets(self.coords[[current]]) 
            
        if path is not None:
            p_coords = self.coords[np.array(path, dtype=int)]
            self.path_line.set_data(p_coords[:, 0], p_coords[:, 1])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(pause_time)
        
        # Giữ cửa sổ hiển thị khi đã tìm thấy đường
        if path is not None:
            plt.ioff()
            plt.show()