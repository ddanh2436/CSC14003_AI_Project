from utils.metrics import run_experiment, measure_memory, run_scalability_test
from utils.visualization import plot_convergence
import matplotlib.pyplot as plt

def run_suite(problem_list, algorithm_configs, n_runs=10):
    """
    Chạy một bộ test (Test Suite) gồm nhiều bài toán.
    
    Args:
        problem_list: Danh sách các bài toán (đã khởi tạo). VD: [Sphere(10), Rastrigin(10)]
        algorithm_configs: Danh sách cấu hình thuật toán. 
                           Dạng: [{'class': HillClimbing, 'params': {...}}, ...]
        n_runs: Số lần chạy mỗi thuật toán để lấy thống kê.
    """
    print("\n" + "="*60)
    print(f"🚀 STARTING TEST SUITE ({len(problem_list)} Problems, {len(algorithm_configs)} Algorithms)")
    print("="*60)

    for problem in problem_list:
        print(f"\n📌 PROBLEM: {problem.name}")
        print("-" * 40)
        
        histories = {} # Để lưu dữ liệu vẽ biểu đồ
        
        # 1. Chạy từng thuật toán trên bài toán này
        for algo_conf in algorithm_configs:
            AlgoClass = algo_conf['class']
            params = algo_conf.get('params', {})
            
            # A. Chạy thống kê (Robustness)
            # Hàm run_experiment đã tự in báo cáo ra màn hình rồi
            stats = run_experiment(AlgoClass, problem, n_runs=n_runs, **params)
            
            # B. Chạy 1 lần nữa để lấy lịch sử vẽ biểu đồ (Convergence Plot)
            # (Chúng ta chạy riêng để đảm bảo biểu đồ thể hiện một lần chạy điển hình)
            opt = AlgoClass(problem, **params)
            opt.solve()
            histories[AlgoClass.__name__] = opt.history

        # 2. Vẽ biểu đồ so sánh ngay sau khi xong 1 bài toán
        print(f"   >> Vẽ biểu đồ so sánh cho {problem.name}...")
        plot_convergence(histories, title=f"Comparison on {problem.name}")
        
    print("\n✅ TEST SUITE COMPLETED!")