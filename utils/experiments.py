from utils.metrics import run_experiment, measure_memory, run_scalability_test
from utils.visualization import plot_convergence
import matplotlib.pyplot as plt

def run_suite(problem_list, algorithm_configs, n_runs=10):
    print("\n" + "="*60)
    print(f"🚀 STARTING TEST SUITE ({len(problem_list)} Problems, {len(algorithm_configs)} Algorithms)")
    print("="*60)

    for problem in problem_list:
        print(f"\n📌 PROBLEM: {problem.name}")
        print("-" * 40)
        
        histories = {}
        times = {} # THÊM MỚI: Khởi tạo từ điển lưu thời gian
        
        for algo_conf in algorithm_configs:
            AlgoClass = algo_conf['class']
            params = algo_conf.get('params', {})
            
            stats = run_experiment(AlgoClass, problem, n_runs=n_runs, **params)
            
            opt = AlgoClass(problem, **params)
            opt.solve()
            histories[AlgoClass.__name__] = opt.history
            times[AlgoClass.__name__] = opt.run_time # THÊM MỚI: Lưu số giây chạy

        print(f"   >> Vẽ biểu đồ so sánh cho {problem.name}...")
        # SỬA ĐỔI: Truyền thêm times_dict vào hàm vẽ biểu đồ
        plot_convergence(histories, title=f"Comparison on {problem.name}", times_dict=times)
        
    print("\n✅ TEST SUITE COMPLETED!")