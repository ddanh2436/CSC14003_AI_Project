import numpy as np

class ContinuousProblem:
    def __init__(self, bounds, name="Continuous Problem"):
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.name = name

    def fitness(self, x):
        raise NotImplementedError

# --- Bài 1: Sphere (Dễ, Lồi) ---
class Sphere(ContinuousProblem):
    def __init__(self, dim=10):
        super().__init__([[-5.12, 5.12]] * dim, name=f"Sphere (D={dim})")

    def fitness(self, x):
        return np.sum(x**2)

# --- Bài 2: Rastrigin (Khó, Đa cực trị) ---
class Rastrigin(ContinuousProblem):
    def __init__(self, dim=10):
        super().__init__([[-5.12, 5.12]] * dim, name=f"Rastrigin (D={dim})")

    def fitness(self, x):
        A = 10
        return A * self.dim + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# --- Bài 3: Rosenbrock (Thung lũng hẹp - Rất khó hội tụ) ---
class Rosenbrock(ContinuousProblem):
    def __init__(self, dim=10):
        super().__init__([[-5, 10]] * dim, name=f"Rosenbrock (D={dim})")

    def fitness(self, x):
        # f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
        return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

# --- Bài 4: Ackley (Nhiều bẫy nhỏ nhưng có phễu lớn) ---
class Ackley(ContinuousProblem):
    def __init__(self, dim=10):
        super().__init__([[-32.768, 32.768]] * dim, name=f"Ackley (D={dim})")

    def fitness(self, x):
        a, b, c = 20, 0.2, 2 * np.pi
        term1 = -a * np.exp(-b * np.sqrt(np.mean(x**2)))
        term2 = -np.exp(np.mean(np.cos(c * x)))
        return term1 + term2 + a + np.exp(1)