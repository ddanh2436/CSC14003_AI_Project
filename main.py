import numpy as np
# Import các bài toán
from problems.continuous import Sphere, Rastrigin, Rosenbrock, Ackley
from problems.discrete import TSP

# Import thuật toán Classical
from algorithms.classical.hill_climbing import HillClimbing
from algorithms.classical.hill_climbing_tsp import HillClimbingTSP
from algorithms.classical.bfs import BreadthFirstSearch
from algorithms.classical.a_star import AStarSearch

# Import thuật toán Evolutionary
from algorithms.evolutionary.ga import GeneticAlgorithm
from algorithms.evolutionary.de import DifferentialEvolution
from algorithms.evolutionary.es import EvolutionStrategy

# Import thuật toán Physics-based
from algorithms.physics.simulated_annealing import SimulatedAnnealing
from algorithms.physics.gsa import GravitationalSearchAlgorithm
from algorithms.physics.hs import HarmonySearch

# Import thuật toán Swarm Intelligence
from algorithms.swarm import PSO, ABC, FA, CS, ACO
from algorithms.human_based.tlbo import TLBO
# Import công cụ thực nghiệm và trực quan hóa
from utils.experiments import run_suite
from utils.metrics import run_scalability_test
from utils.visualization import plot_3d_surface

# Thử nghiệm import module phân tích nâng cao
try:
    from utils.sensitivity import run_parameter_sweep, plot_sensitivity_results
    from utils.advanced_analysis import plot_exploration_exploitation
    HAS_ADVANCED_TOOLS = True
except ImportError:
    HAS_ADVANCED_TOOLS = False

def main():
    # ==========================================
    # SCENARIO 0: TRỰC QUAN HÓA ĐỊA HÌNH 3D (2D)
    # ==========================================
    print("\n" + "="*60)
    print("SCENARIO 0: 3D LANDSCAPE VISUALIZATION")
    print("="*60)
    
    viz_problems = [Sphere(dim=2), Rastrigin(dim=2), Ackley(dim=2)]
    for prob in viz_problems:
        print(f"🎨 Đang vẽ 3D landscape cho: {prob.name}")
        plot_3d_surface(prob, title=f"Landscape of {prob.name}")

    # ==========================================
    # SCENARIO 1: BENCHMARK BÀI TOÁN LIÊN TỤC (10D)
    # ==========================================
    print("\n" + "="*60)
    print("SCENARIO 1: CONTINUOUS OPTIMIZATION BENCHMARK (10D)")
    print("="*60)
    
    continuous_problems = [
        Sphere(dim=10),      # Unimodal
        Rastrigin(dim=10),   # Multimodal
        Rosenbrock(dim=10),  # Narrow valley
        Ackley(dim=10)       # Local optima traps
    ]
    
    continuous_algos = [
        {'class': HillClimbing, 'params': {'max_iter': 500, 'step_size': 0.5}},
        {'class': SimulatedAnnealing, 'params': {'max_iter': 500, 'step_size': 0.5}},
        {'class': GravitationalSearchAlgorithm, 'params': {'max_iter': 100, 'pop_size': 20}},
        {'class': HarmonySearch, 'params': {'max_iter': 500, 'pop_size': 20}},
        {'class': GeneticAlgorithm, 'params': {'max_iter': 500, 'pop_size': 50, 'mutation_rate': 0.1}},
        {'class': DifferentialEvolution, 'params': {'max_iter': 500, 'pop_size': 50}},
        {'class': EvolutionStrategy, 'params': {'max_iter': 500, 'pop_size': 20}},
        {'class': PSO, 'params': {'max_iter': 500, 'pop_size': 30}},
        {'class': ABC, 'params': {'max_iter': 500, 'pop_size': 40}},
        {'class': CS, 'params': {'max_iter': 500, 'pop_size': 25}},
        {'class': TLBO, 'params': {'max_iter': 500, 'pop_size': 40}}
    ]
    
    # Chạy thực nghiệm lấy thống kê Robustness (n_runs=10)
    run_suite(continuous_problems, continuous_algos, n_runs=10)

    # ==========================================
    # PHÂN TÍCH KHẢ NĂNG MỞ RỘNG (SCALABILITY)
    # ==========================================
    print("\n" + "="*60)
    print("SCENARIO: SCALABILITY ANALYSIS (Time vs Dimension)")
    print("="*60)
    run_scalability_test(
        [PSO, GeneticAlgorithm, DifferentialEvolution], 
        Sphere, 
        dims=[2, 10, 30, 50]
    )

    # ==========================================
    # PHÂN TÍCH NÂNG CAO (SENSITIVITY & EXPLORATION)
    # ==========================================
    if HAS_ADVANCED_TOOLS:
        print("\n" + "="*60)
        print("SCENARIO 3: PARAMETER SENSITIVITY & DIVERSITY ANALYSIS")
        print("="*60)
        
        # 1. Phân tích độ nhạy tham số cho GA
        ga_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
        prob_rast = Rastrigin(dim=10)
        
        df_ga = run_parameter_sweep(
            GeneticAlgorithm, prob_rast, 
            param_name='mutation_rate', 
            values=ga_values, 
            n_runs=10, max_iter=200, pop_size=50
        )
        plot_sensitivity_results(df_ga, 'mutation_rate', 'GA', 'Rastrigin')

        # 2. Minh họa Exploration vs Exploitation cho PSO
        print("🔍 Đang phân tích hành vi Exploration vs Exploitation cho PSO...")
        pso_test = PSO(Sphere(dim=10), max_iter=200, pop_size=30)
        pso_test.solve()
        plot_exploration_exploitation(pso_test.diversity_history, pso_test.history, title="PSO Behavior Analysis")

    # ==========================================
    # SCENARIO 2: BÀI TOÁN RỜI RẠC (TSP)
    # ==========================================
    print("\n" + "="*60)
    print("SCENARIO 2: DISCRETE OPTIMIZATION (TSP)")
    print("="*60)
    
    discrete_problems = [TSP(n_cities=20)]
    discrete_algos = [
        {'class': HillClimbingTSP, 'params': {'max_iter': 2000}},
        {'class': ACO, 'params': {'max_iter': 100, 'n_ants': 20, 'decay': 0.5}}
    ]
    
    run_suite(discrete_problems, discrete_algos, n_runs=5)

if __name__ == "__main__":
    main()