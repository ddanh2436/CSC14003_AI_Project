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

    # ==========================================
# 2. KNAPSACK PROBLEM (CÁI TÚI)
# ==========================================
class Knapsack(DiscreteProblem):
    def __init__(self, n_items=50, capacity=None, seed=42):
        super().__init__(name=f"Knapsack ({n_items} items)")
        self.n_items = n_items
        self.dim = n_items # Số chiều = số món đồ (0 hoặc 1)
        
        np.random.seed(seed)
        # Random trọng lượng (1-10) và giá trị (10-100)
        self.weights = np.random.randint(1, 15, n_items)
        self.values = np.random.randint(10, 100, n_items)
        
        # Nếu không quy định sức chứa, mặc định là 50% tổng trọng lượng
        if capacity is None:
            self.capacity = int(np.sum(self.weights) * 0.5)
        else:
            self.capacity = capacity

    def fitness(self, solution):
        selected = np.round(solution).astype(int)
        
        total_weight = np.sum(selected * self.weights)
        
        # --- GREEDY REPAIR TĂNG CƯỜNG ---
        if total_weight > self.capacity:
            # Tính tỷ lệ Giá trị / Trọng lượng
            ratios = self.values / self.weights
            
            # Lấy index của các món đồ đang được chọn
            selected_indices = np.where(selected == 1)[0]
            
            # Sắp xếp các món đồ ĐANG CHỌN theo tỷ lệ tăng dần (món "tồi" lên đầu)
            sorted_by_ratio = sorted(selected_indices, key=lambda idx: ratios[idx])
            
            # Vứt đồ ra khỏi túi cho đến khi đủ cân
            for idx in sorted_by_ratio:
                selected[idx] = 0
                total_weight -= self.weights[idx]
                if total_weight <= self.capacity:
                    break
                    
        # Tính lại value sau khi đã repair
        total_value = np.sum(selected * self.values)
        return -total_value

# ==========================================
# 3. SHORTEST PATH (GRAPH)
# ==========================================
class ShortestPath(DiscreteProblem):
    def __init__(self, n_nodes=50, edge_prob=0.3, seed=42):
        super().__init__(name=f"Shortest Path ({n_nodes} nodes)")
        self.n_nodes = n_nodes
        self.start_node = 0
        self.goal_node = n_nodes - 1
        
        np.random.seed(seed)
        # Thêm tọa độ cho đồ thị (để tính Heuristic cho thuật toán A*)
        self.coords = np.random.rand(n_nodes, 2) * 100
        
        # Đảm bảo start và goal cách xa nhau một chút
        self.coords[self.start_node] = [0, 0]
        self.coords[self.goal_node] = [100, 100]

        self.adj_matrix = np.zeros((n_nodes, n_nodes))
        
        # Tạo cạnh ngẫu nhiên và tính trọng số dựa trên tọa độ thực tế
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if np.random.rand() < edge_prob:
                    # Trọng số chính là khoảng cách Euclidean thực tế
                    dist = np.linalg.norm(self.coords[i] - self.coords[j])
                    self.adj_matrix[i][j] = dist
                    self.adj_matrix[j][i] = dist # Đồ thị vô hướng

    def get_neighbors(self, node):
        """Hàm hỗ trợ cho BFS/A* lấy danh sách hàng xóm"""
        return np.where(self.adj_matrix[node] > 0)[0]

    def get_cost(self, u, v):
        """Lấy trọng số cạnh u-v"""
        return self.adj_matrix[u][v]

    def heuristic(self, node):
        """
        Hàm Heuristic dành riêng cho A* Search.
        Ước lượng khoảng cách đường chim bay (Euclidean) từ node hiện tại tới đích.
        """
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