import numpy as np
from algorithms.optimizer import Optimizer

class AntColonyOptimization(Optimizer):
    def __init__(self, problem, n_ants=10, decay=0.5, alpha=1, beta=2, **kwargs):
        super().__init__(problem, **kwargs)
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
        # Lấy dữ liệu từ bài toán TSP
        self.n_cities = self.problem.n_cities
        self.dist_matrix = self.problem.dist_matrix
        
        # Khởi tạo Pheromone
        self.pheromone = np.ones((self.n_cities, self.n_cities)) / self.n_cities

    def _evolve(self):
        # Không có best solution ban đầu, ta chạy vòng lặp luôn
        self.save_history() # Lưu giá trị inf ban đầu (hoặc bạn có thể chạy thử 1 con kiến để init)

        for _ in range(self.max_iter):
            all_paths = []
            all_path_lens = []
            
            # Mỗi kiến xây dựng 1 đường đi
            for _ in range(self.n_ants):
                path = self._construct_path()
                path_len = self.problem.fitness(path) # Dùng hàm fitness của bài toán
                all_paths.append(path)
                all_path_lens.append(path_len)
                
                # Cập nhật Global Best nếu tìm thấy đường tốt hơn
                if path_len < self.global_best_fitness:
                    self.update_global_best(np.array(path), path_len)
            
            # Cập nhật Pheromone
            self.pheromone *= (1 - self.decay) # Bay hơi
            
            for path, length in zip(all_paths, all_path_lens):
                for j in range(len(path) - 1):
                    self.pheromone[path[j], path[j+1]] += 1.0 / (length + 1e-10)
                # Đoạn khép kín vòng
                self.pheromone[path[-1], path[0]] += 1.0 / (length + 1e-10)

            self.save_history()

        return self.global_best_solution, self.global_best_fitness

    def _construct_path(self):
        start_node = np.random.randint(0, self.n_cities)
        path = [start_node]
        visited = set(path)
        
        for _ in range(self.n_cities - 1):
            curr = path[-1]
            probs = self._calculate_probs(curr, visited)
            next_city = self._roulette_wheel_selection(probs)
            path.append(next_city)
            visited.add(next_city)
        return path

    def _calculate_probs(self, curr, visited):
        probs = np.zeros(self.n_cities)
        for city in range(self.n_cities):
            if city not in visited:
                heuristic = 1.0 / (self.dist_matrix[curr][city] + 1e-10)
                probs[city] = (self.pheromone[curr][city] ** self.alpha) * (heuristic ** self.beta)
        
        total = np.sum(probs)
        return probs / (total + 1e-10)

    def _roulette_wheel_selection(self, probs):
        cumulative_probs = np.cumsum(probs)
        r = np.random.rand()
        return np.searchsorted(cumulative_probs, r)