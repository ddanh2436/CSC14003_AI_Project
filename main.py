# Import bài toán
from problems.continuous import Sphere, Rastrigin, Rosenbrock, Ackley
from problems.discrete import TSP # (File này chúng ta sẽ tạo ở bước sau)

# Import thuật toán
from algorithms.classical.hill_climbing import HillClimbing
from algorithms.evolutionary.ga import GeneticAlgorithm
from algorithms.physics.simulated_annealing import SimulatedAnnealing
from algorithms.swarm import PSO, ABC, FA, CS, ACO
# from algorithms.classical.hill_climbing_tsp import HillClimbingTSP (Sẽ dùng cho bài toán rời rạc)

from algorithms.physics.gsa import GravitationalSearchAlgorithm
from algorithms.physics.hs import HarmonySearch
# Import công cụ chạy
from utils.experiments import run_suite

def main():
    # ==========================================
    # KỊCH BẢN 1: BÀI TOÁN LIÊN TỤC (CONTINUOUS)
    # ==========================================
    print("\n==============================================")
    print("SCENARIO 1: CONTINUOUS OPTIMIZATION BENCHMARK")
    print("==============================================")
    
    # 1. Chọn các bài test (Test Cases)
    # Bạn muốn so sánh trên nhiều địa hình khác nhau
    continuous_problems = [
        Sphere(dim=10),      # Dễ, lồi
        Rastrigin(dim=10),   # Khó, đa cực trị (nhiều đỉnh nhọn)
        # Ackley(dim=10)     # (Mở comment nếu muốn test thêm)
    ]
    
    # 2. Chọn các thuật toán và tham số (Algorithm Configs)
    continuous_algos = [
        {
            'class': HillClimbing, 
            'params': {'max_iter': 500, 'step_size': 0.5}
        },
        {
            'class': SimulatedAnnealing,
            'params': {'max_iter': 500, 'step_size': 0.5, 'initial_temp': 1000}
        },
        {
            'class': GeneticAlgorithm,
            'params': {'max_iter': 500, 'pop_size': 50, 'mutation_rate': 0.1}
        },
        {
            'class': GravitationalSearchAlgorithm,
            'params': {'max_iter': 500, 'pop_size': 40, 'G0': 100, 'alpha': 20}
        },
        # Thêm Harmony Search
        {
            'class': HarmonySearch,
            'params': {'max_iter': 500, 'pop_size': 20, 'hmcr': 0.95, 'par': 0.3}
        }
    ]
    
    # 3. Kích hoạt chạy bộ test
    run_suite(continuous_problems, continuous_algos, n_runs=30)


    # ==========================================
    # KỊCH BẢN 2: BÀI TOÁN RỜI RẠC (DISCRETE)
    # ==========================================
    # Hiện tại chúng ta chưa cài xong thuật toán cho TSP (GA/SA cần sửa đổi cho TSP)
    # Nên tôi tạm comment lại để code không lỗi. 
    # Khi bạn cài xong HillClimbingTSP ở bước trước thì mở ra nhé.
    
    """
    print("\n==============================================")
    print("SCENARIO 2: DISCRETE OPTIMIZATION (TSP)")
    print("==============================================")
    
    discrete_problems = [
        TSP(n_cities=10),
        TSP(n_cities=20)
    ]
    
    discrete_algos = [
        {
            'class': HillClimbingTSP,
            'params': {'max_iter': 1000}
        }
        # Sau này thêm GA_TSP, SA_TSP vào đây
    ]
    
    run_suite(discrete_problems, discrete_algos, n_runs=5)
    """

if __name__ == "__main__":
    main()