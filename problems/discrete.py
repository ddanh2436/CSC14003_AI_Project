import numpy as np
import matplotlib.pyplot as plt

class DiscreteProblem:
    """Class cha cho các bài toán rời rạc"""
    def __init__(self, name="Discrete Problem"):
        self.name = name
        # Bài toán rời rạc không có bounds liên tục như [-5, 5]
        self.bounds = None 
        self.dim = 0

    def fitness(self, solution):
        raise NotImplementedError

class GraphProblem(DiscreteProblem):
    """
    Bài toán tìm đường đi trên đồ thị dành cho các thuật toán cổ điển
    - adjacency_list: { node: {neighbor: weight} }
    - heuristic: { node: h_value } (dành cho A* và GBF)
    """
    def __init__(self, start_node, goal_node, adjacency_list=None, heuristic=None):
        self.start_node = start_node
        self.goal_node = goal_node
        self.adjacency_list = adjacency_list if adjacency_list else {}
        self.heuristic = heuristic if heuristic else {}
    def get_neighbors(self, node):
        #Trả về các láng giếng của node được chọn
        return list(self.adjacency_list.get(node, {}).key())
    def get_cost(self, from_node, to_node):
        # Trả về giá trị của hàm g(n) (trọng số) từ node này sang node khác
        return self.adjacency_list.get(from_node, {}).get(to_node, float('inf'))
    def heuristic(self, node):
        # Trả về giá trị hàm h(n) từ node đến goal:
        return self.heuristic.get(node, 0)
    def fitness(self, path):
        if not path or path[0] != self.start_node or path[-1] != self.goal_node:
            return float('inf')
        
        total_cost = 0
        for i in range (len(path) - 1):
            cost = self.get_cost(path[i], path[i + 1])
            if cost == float('inf'): return float('inf')
            total_cost += cost
        return total_cost
    def create_sample_graph():
        """
        Tạo đồ thị mẫu tương tự ví dụ trong sách giáo khoa (ví dụ: Map of Romania).
        """
        adj = {
            'A': {'B': 2, 'C': 5},
            'B': {'A': 2, 'D': 2, 'E': 4},
            'C': {'A': 5, 'G': 10},
            'D': {'B': 2, 'G': 5},
            'E': {'B': 4, 'G': 1},
            'G': {'C': 10, 'D': 5, 'E': 1}
        }
        
        # Giả định Goal là 'G', heuristics là khoảng cách đường chim bay đến G
        h = {
            'A': 6, 'B': 5, 'C': 7, 'D': 3, 'E': 1, 'G': 0
        }
        
        return GraphProblem(start_node='A', goal_node='G', adjacency_list=adj, heuristics=h)
class TSP(DiscreteProblem):
    """
    Traveling Salesman Problem (TSP) - Bài toán người du lịch
    Mục tiêu: Tìm lộ trình đi qua tất cả thành phố rồi quay về điểm đầu sao cho tổng quãng đường ngắn nhất.
    """
    def __init__(self, n_cities=20, seed=42):
        super().__init__(name=f"TSP ({n_cities} cities)")
        self.n_cities = n_cities
        self.dim = n_cities # Số chiều = Số thành phố
        
        # Cố định seed để mỗi lần chạy đều ra bản đồ giống nhau (dễ so sánh)
        np.random.seed(seed)
        
        # Tạo toạ độ ngẫu nhiên cho các thành phố (x, y) trong khoảng [0, 100]
        self.cities = np.random.rand(n_cities, 2) * 100
        
        # Tính trước ma trận khoảng cách (Distance Matrix) để thuật toán chạy nhanh hơn
        # Thay vì tính lại khoảng cách mỗi lần, ta tra bảng
        self.dist_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                # Khoảng cách Euclidean: sqrt((x1-x2)^2 + (y1-y2)^2)
                self.dist_matrix[i, j] = np.linalg.norm(self.cities[i] - self.cities[j])

    def fitness(self, path):
        """
        Tính tổng độ dài quãng đường của lộ trình (path).
        Path là danh sách chỉ số thành phố, ví dụ: [0, 5, 2, 9...]
        """
        # Đảm bảo path là kiểu số nguyên
        path = np.array(path, dtype=int)
        
        total_dist = 0
        # Cộng khoảng cách giữa các thành phố liên tiếp
        for i in range(len(path) - 1):
            total_dist += self.dist_matrix[path[i], path[i+1]]
            
        # Cộng khoảng cách từ điểm cuối quay về điểm đầu
        total_dist += self.dist_matrix[path[-1], path[0]]
        
        return total_dist

    def visualize(self, path, title="TSP Route"):
        """Vẽ bản đồ và đường đi"""
        plt.figure(figsize=(8, 6))
        
        # 1. Vẽ các điểm thành phố (chấm đỏ)
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=50, zorder=2, label='Cities')
        
        # 2. Đánh số thứ tự thành phố
        for i, (x, y) in enumerate(self.cities):
            plt.text(x + 1, y + 1, str(i), fontsize=9)
            
        # 3. Vẽ đường nối (màu xanh)
        path = np.array(path, dtype=int)
        # Thêm điểm đầu vào cuối path để vẽ đường khép kín
        closed_path = np.append(path, path[0])
        
        route_coords = self.cities[closed_path]
        plt.plot(route_coords[:, 0], route_coords[:, 1], c='blue', linestyle='-', linewidth=1, zorder=1, alpha=0.7)
        
        plt.title(f"{title}\nTotal Distance: {self.fitness(path):.2f}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()