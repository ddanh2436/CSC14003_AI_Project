import numpy as np

class ContinuousProblem:
    def __init__(self, bounds, name="Continuous Problem"):
        self.bounds = np.array(bounds) # Ma trận giới hạn [[min, max], [min, max],...]
        self.dim = len(bounds)         # Số chiều (Dimension)
        self.name = name

    def fitness(self, x):
        raise NotImplementedError

# --- Bài toán 1: Hàm Sphere (Dễ, Lồi) ---
# f(x) = sum(x^2). Min = 0 tại [0,0,...]
class Sphere(ContinuousProblem):
    def __init__(self, dim=10):
        # Giới hạn tìm kiếm thường là [-5.12, 5.12]
        bounds = [[-5.12, 5.12]] * dim
        super().__init__(bounds, name="Sphere Function")

    def fitness(self, x):
        return np.sum(x**2)

# --- Bài toán 2: Hàm Rastrigin (Khó, Nhiều cực trị địa phương) ---
# Dùng để test khả năng thoát khỏi bẫy cục bộ của thuật toán
class Rastrigin(ContinuousProblem):
    def __init__(self, dim=10):
        bounds = [[-5.12, 5.12]] * dim
        super().__init__(bounds, name="Rastrigin Function")

    def fitness(self, x):
        A = 10
        # Công thức Rastrigin chuẩn
        return A * self.dim + np.sum(x**2 - A * np.cos(2 * np.pi * x))