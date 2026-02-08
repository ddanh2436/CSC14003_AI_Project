# Import bài toán
from problems.continuous import Sphere, Rastrigin, Rosenbrock, Ackley
from problems.discrete import TSP

# Import thuật toán Classical
from algorithms.classical.hill_climbing import HillClimbing
from algorithms.classical.hill_climbing_tsp import HillClimbingTSP
from algorithms.classical.bfs import BreadthFirstSearch
from algorithms.classical.a_star import AStarSearch

# Import thuật toán Evolutionary (Đã thêm DE và ES)
from algorithms.evolutionary.ga import GeneticAlgorithm
# Đảm bảo bạn đã tạo file de.py và es.py trong thư mục algorithms/evolutionary/
from algorithms.evolutionary.de import DifferentialEvolution 
from algorithms.evolutionary.es import EvolutionStrategy 

# Import thuật toán Physics-based (Đường dẫn đúng cho SimulatedAnnealing)
from algorithms.physics.simulated_annealing import SimulatedAnnealing
from algorithms.physics.gsa import GravitationalSearchAlgorithm
from algorithms.physics.hs import HarmonySearch

# Import thuật toán Swarm/Biology-based
from algorithms.swarm import PSO, ABC, FA, CS, ACO

# Import công cụ chạy
from utils.experiments import run_suite
from utils.metrics import run_scalability_test

def main():
    # ==========================================
    # KỊCH BẢN 1: BÀI TOÁN LIÊN TỤC (CONTINUOUS)
    # ==========================================
    print("\n" + "="*60)
    print("SCENARIO 1: CONTINUOUS OPTIMIZATION BENCHMARK")
    print("="*60)
    
    continuous_problems = [
        Sphere(dim=10),      # Dễ, lồi
        Rastrigin(dim=10),   # Khó, đa cực trị
        Rosenbrock(dim=10),  # Thung lũng hẹp
        Ackley(dim=10)       # Nhiều local optima
    ]
    
    continuous_algos = [
        {'class': HillClimbing, 'params': {'max_iter': 500, 'step_size': 0.5}},
        {'class': SimulatedAnnealing, 'params': {'max_iter': 500, 'step_size': 0.5, 'initial_temp': 1000}},
        {'class': GravitationalSearchAlgorithm, 'params': {'max_iter': 100, 'pop_size': 20}},
        {'class': HarmonySearch, 'params': {'max_iter': 500, 'pop_size': 20}},
        {'class': GeneticAlgorithm, 'params': {'max_iter': 500, 'pop_size': 50, 'mutation_rate': 0.1}},
        {'class': DifferentialEvolution, 'params': {'max_iter': 500, 'pop_size': 50}},
        {'class': EvolutionStrategy, 'params': {'max_iter': 500, 'pop_size': 20}},
        {'class': PSO, 'params': {'max_iter': 500, 'pop_size': 30}},
        {'class': ABC, 'params': {'max_iter': 500, 'pop_size': 50}},
        {'class': CS, 'params': {'max_iter': 500, 'pop_size': 25}}
    ]
    
    run_suite(continuous_problems, continuous_algos, n_runs=30)

    # 4. Chạy kiểm tra khả năng mở rộng (Scalability)
    print("\n--- Performing Scalability Analysis ---")
    run_scalability_test([PSO, GeneticAlgorithm, DifferentialEvolution], Sphere, dims=[2, 10, 30, 50])

    # ==========================================
    # KỊCH BẢN 2: BÀI TOÁN RỜI RẠC (DISCRETE - TSP)
    # ==========================================
    print("\n" + "="*60)
    print("SCENARIO 2: DISCRETE OPTIMIZATION (TSP)")
    print("="*60)
    
    discrete_problems = [TSP(n_cities=20)]
    discrete_algos = [
        {'class': HillClimbingTSP, 'params': {'max_iter': 2000}},
        {'class': ACO, 'params': {'max_iter': 100, 'n_ants': 20}}
    ]
    
    run_suite(discrete_problems, discrete_algos, n_runs=5)

if __name__ == "__main__":
    main()