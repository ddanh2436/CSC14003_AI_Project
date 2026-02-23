import numpy as np
from scipy import stats  # Thêm thư viện thống kê cho Scenario 4

from problems.continuous import Sphere, Rastrigin, Rosenbrock, Ackley
from problems.discrete import TSP, Knapsack, ShortestPath

from algorithms.classical.hill_climbing import HillClimbing
from algorithms.classical.hill_climbing_tsp import HillClimbingTSP
from algorithms.classical.hill_climbing_knapsack import HillClimbingKnapsack
from algorithms.classical.bfs import BreadthFirstSearch
from algorithms.classical.a_star import AStarSearch
from algorithms.classical.dfs import DepthFirstSearch
from algorithms.classical.ucs import UniformCostSearch
from algorithms.classical.gbf import GreedyBestFirstSearch

# Nhóm Evolutionary
from algorithms.evolutionary.ga import GeneticAlgorithm
from algorithms.evolutionary.ga_tsp import GeneticAlgorithmTSP  # Thuật toán mới cho TSP
from algorithms.evolutionary.de import DifferentialEvolution
from algorithms.evolutionary.es import EvolutionStrategy

# Nhóm Physics-based
from algorithms.physics.simulated_annealing import SimulatedAnnealing
from algorithms.physics.sa_tsp import SimulatedAnnealingTSP  # Thuật toán mới cho TSP
from algorithms.physics.sa_knapsack import SimulatedAnnealingKnapsack #Thuật toán mới cho Knapsack
from algorithms.physics.gsa import GravitationalSearchAlgorithm
from algorithms.physics.hs import HarmonySearch

# Nhóm Swarm Intelligence & Human-based
from algorithms.swarm import PSO, ABC, FA, CS, ACO
from algorithms.human_based.tlbo import TLBO

# Công cụ thực nghiệm và trực quan hóa
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
    # SCENARIO 2: BÀI TOÁN RỜI RẠC (DISCRETE)
    # ==========================================
    
    # --- 2.1 TRAVELING SALESMAN PROBLEM (TSP) ---
    print("\n" + "-"*60)
    print("SCENARIO 2.1: TRAVELING SALESMAN PROBLEM (TSP)")
    print("-"*60)
    
    tsp_prob = TSP(n_cities=20)
    tsp_problems = [tsp_prob]
    
    tsp_algos = [
        # Sử dụng các bản chuyên dụng cho TSP
        {'class': HillClimbingTSP, 'params': {'max_iter': 2000}},
        {'class': SimulatedAnnealingTSP, 'params': {'max_iter': 2000, 'initial_temp': 1000, 'cooling_rate': 0.95}},
        {'class': GeneticAlgorithmTSP, 'params': {'max_iter': 500, 'pop_size': 50}},
        {'class': ACO, 'params': {'max_iter': 200, 'n_ants': 20, 'decay': 0.5}}
    ]
    
    # Chạy thống kê so sánh
    run_suite(tsp_problems, tsp_algos, n_runs=5)

    # TRỰC QUAN HÓA: Vẽ bản đồ TSP tự động
    print("\n🗺️ Đang vẽ bản đồ lộ trình tốt nhất cho TSP...")
    aco_solver = ACO(tsp_prob, max_iter=300, n_ants=30)
    best_path, best_dist, _ = aco_solver.solve()
    tsp_prob.visualize(best_path, title=f"Best TSP Route (ACO) - Dist: {best_dist:.2f}")

    # --- 2.2 KNAPSACK PROBLEM (CÁI TÚI) ---
    print("\n" + "-"*60)
    print("SCENARIO 2.2: KNAPSACK PROBLEM (0/1 Selection)")
    print("-"*60)

    knapsack_problems = [Knapsack(n_items=50)]
    knapsack_algos = [
        # Baseline cổ điển
        {'class': HillClimbingKnapsack, 'params': {'max_iter': 500}},
        # Cổ điển có thể thoát bẫy
        {'class': SimulatedAnnealingKnapsack, 'params': {'max_iter': 500, 'initial_temp': 1000}},
        # Tiến hóa (Hoạt động tốt nhất trên chuỗi nhị phân)
        {'class': GeneticAlgorithm, 'params': {'max_iter': 500, 'pop_size': 50}},
        # Swarm (Ong mật - dùng cơ chế làm tròn ngầm)
        {'class': ABC, 'params': {'max_iter': 500, 'pop_size': 40}}
    ]
    run_suite(knapsack_problems, knapsack_algos, n_runs=5)
   
    # --- 2.3 SHORTEST PATH (GRAPH SEARCH) ---
    print("\n" + "-"*60)
    print("SCENARIO 2.3: SHORTEST PATH (Graph Traversal)")
    print("-"*60)

    # 1. Chạy đánh giá hiệu năng (Benchmark Bar Chart) trên đồ thị lớn
    graph_problems = [ShortestPath(n_nodes=2000, edge_prob=0.2)]
    graph_algos = [
        {'class': DepthFirstSearch, 'params': {}},
        {'class': BreadthFirstSearch, 'params': {}},
        {'class': UniformCostSearch, 'params': {}},
        {'class': GreedyBestFirstSearch, 'params': {}},
        {'class': AStarSearch, 'params': {}}
    ]
    run_suite(graph_problems, graph_algos, n_runs=1)

    # Khởi tạo đồ thị 70 node dùng chung cho cả phần Vẽ tĩnh và Animation
    viz_graph_prob = ShortestPath(n_nodes=70, edge_prob=0.15)

    # 2. Thu thập và vẽ bản đồ so sánh đa luồng (Static Multi-path Plot)
    print("\n🗺️ Đang vẽ bản đồ tĩnh so sánh lộ trình của 5 thuật toán...")
    paths_dict = {}
    for algo_conf in graph_algos:
        AlgoClass = algo_conf['class']
        opt = AlgoClass(viz_graph_prob, **algo_conf['params'])
        best_path, _, _ = opt.solve()
        if best_path is not None:
            paths_dict[AlgoClass.__name__] = best_path
            
    # Gọi hàm vẽ đè các đường đi (Hàm visualize_paths vừa thêm ở file discrete.py)
    viz_graph_prob.visualize_paths(paths_dict, title=f"Path Tracing Comparison ({viz_graph_prob.n_nodes} nodes)")

    # 3. Trình diễn Animation (BFS và A*)
    print("\n🎬 TRÌNH DIỄN THUẬT TOÁN BFS & A* (ANIMATION)...")
    
    print(">> Đang chạy: A* Search...")
    astar_viz = AStarSearch(viz_graph_prob, visualize=True, pause_time=0.03)
    astar_viz.solve()
    
    # Bạn có thể đóng cửa sổ A* lại, chương trình sẽ tiếp tục bật animation của BFS
    print(">> Đang chạy: BFS Search...")
    bfs_viz = BreadthFirstSearch(viz_graph_prob, visualize=True, pause_time=0.03)
    bfs_viz.solve()

    # ==========================================
    # SCENARIO 3: SCALABILITY ANALYSIS
    # ==========================================
    print("\n" + "="*60)
    print("SCENARIO 3: SCALABILITY ANALYSIS (Time vs Dimension)")
    print("="*60)
    run_scalability_test(
        [PSO, GeneticAlgorithm, DifferentialEvolution], 
        Sphere, 
        dims=[10, 30, 50]
    )

    # ==========================================
    # SCENARIO 4: STATISTICAL HYPOTHESIS TESTING
    # ==========================================
    print("\n" + "="*60)
    print("SCENARIO 4: STATISTICAL HYPOTHESIS TESTING (T-TEST & WILCOXON)")
    print("="*60)
    
    n_runs_stats = 30
    test_problem = Rastrigin(dim=10)
    
    print(f"Đang chạy thu thập mẫu cho PSO ({n_runs_stats} lần)...")
    pso_results = [PSO(test_problem, max_iter=200, pop_size=30).solve()[1] for _ in range(n_runs_stats)]
        
    print(f"Đang chạy thu thập mẫu cho DE ({n_runs_stats} lần)...")
    de_results = [DifferentialEvolution(test_problem, max_iter=200, pop_size=30).solve()[1] for _ in range(n_runs_stats)]
        
    print(f"\n[KẾT QUẢ TRUNG BÌNH]")
    print(f"PSO: {np.mean(pso_results):.4f} ± {np.std(pso_results):.4f}")
    print(f"DE:  {np.mean(de_results):.4f} ± {np.std(de_results):.4f}")
    
    # Thực hiện kiểm định
    t_stat, p_val_t = stats.ttest_ind(pso_results, de_results)
    w_stat, p_val_w = stats.ranksums(pso_results, de_results)
    
    print("\n[KIỂM ĐỊNH THỐNG KÊ (Alpha = 0.05)]")
    print(f"1. T-Test P-value: {p_val_t:.4e}")
    print(f"2. Wilcoxon P-value: {p_val_w:.4e}")
    
    if p_val_w < 0.05:
        print("=> KẾT LUẬN: P-value < 0.05. Có sự khác biệt CÓ Ý NGHĨA THỐNG KÊ giữa PSO và DE.")
        better_algo = "PSO" if np.mean(pso_results) < np.mean(de_results) else "DE"
        print(f"=> Nhận định: {better_algo} hoạt động tốt hơn hẳn thuật toán còn lại trên bài toán Rastrigin 10D.")
    else:
        print("=> KẾT LUẬN: P-value >= 0.05. KHÔNG CÓ sự khác biệt rõ rệt về mặt thống kê.")

    # ==========================================
    # SCENARIO 5: PHÂN TÍCH NÂNG CAO (SENSITIVITY & EXPLORATION)
    # ==========================================
    if HAS_ADVANCED_TOOLS:
        print("\n" + "="*60)
        print("SCENARIO 5: PARAMETER SENSITIVITY & DIVERSITY ANALYSIS")
        print("="*60)
        
        ga_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
        prob_rast = Rastrigin(dim=10)
        
        df_ga = run_parameter_sweep(
            GeneticAlgorithm, prob_rast, 
            param_name='mutation_rate', 
            values=ga_values, 
            n_runs=10, max_iter=200, pop_size=50
        )
        plot_sensitivity_results(df_ga, 'mutation_rate', 'GA', 'Rastrigin')

        print("🔍 Đang phân tích hành vi Exploration vs Exploitation cho PSO...")
        pso_test = PSO(Sphere(dim=10), max_iter=200, pop_size=30)
        pso_test.solve()
        plot_exploration_exploitation(pso_test.diversity_history, pso_test.history, title="PSO Behavior Analysis")

if __name__ == "__main__":
    main()