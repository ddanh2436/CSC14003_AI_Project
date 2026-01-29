import numpy as np
import time
import tracemalloc
import matplotlib.pyplot as plt

def run_experiment(optimizer_class, problem, n_runs=30, **kwargs):
    """
    Chạy thực nghiệm và in báo cáo dạng rút gọn (One-line summary).
    """
    fitness_results = []
    time_results = []
    
    # In thông báo đang chạy (dùng end="" để không xuống dòng)
    print(f"⏳ Running {optimizer_class.__name__:<16} ({n_runs} runs)... ", end="", flush=True)
    
    for i in range(n_runs):
        # Khởi tạo và chạy thuật toán
        optimizer = optimizer_class(problem, **kwargs)
        _, best_fitness, _ = optimizer.solve()
        
        fitness_results.append(best_fitness)
        time_results.append(optimizer.run_time)

    # Tính toán thống kê
    mean_fit = np.mean(fitness_results)
    std_fit = np.std(fitness_results)
    best_fit = np.min(fitness_results)
    avg_time = np.mean(time_results)
    
    # In kết quả dạng ONE-LINE (Gọn gàng)
    # Ví dụ: ✅ HillClimbing | Fit: 2.50 ± 1.20 | Best: 0.05 | Time: 0.001s
    print(f"Done!")
    print(f"   ✅ {optimizer_class.__name__:<16} | Fit: {mean_fit:10.4f} ± {std_fit:.4f} | Best: {best_fit:10.4f} | Time: {avg_time:.4f}s")
    
    return {
        "algorithm": optimizer_class.__name__,
        "mean_fitness": mean_fit,
        "std_fitness": std_fit,
        "best_fitness": best_fit,
        "avg_time": avg_time
    }

def measure_memory(optimizer_class, problem, **kwargs):
    """Đo bộ nhớ RAM tiêu thụ"""
    tracemalloc.start()
    
    opt = optimizer_class(problem, **kwargs)
    opt.solve()
    
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_mb = peak / (1024 * 1024)
    print(f"   💾 Memory ({optimizer_class.__name__}): {peak_mb:.4f} MB")
    return peak_mb

def run_scalability_test(optimizer_classes, problem_class, dims=[10, 30, 50, 100], **kwargs):
    """
    Test khả năng mở rộng (Scalability) cho NHIỀU thuật toán cùng lúc.
    Args:
        optimizer_classes: Danh sách Class thuật toán (VD: [HillClimbing, GeneticAlgorithm])
        problem_class: Class bài toán
        dims: Các chiều cần test
    """
    print(f"\n📈 Running Scalability Comparison...")
    
    plt.figure(figsize=(10, 6))
    
    # Duyệt qua từng thuật toán trong danh sách
    for opt_class in optimizer_classes:
        times = []
        print(f"   Testing {opt_class.__name__:<16} | Dims: {dims} ... ", end="", flush=True)
        
        for d in dims:
            prob = problem_class(dim=d)
            # Chạy ngầm 3 lần lấy trung bình time cho chính xác
            start = time.time()
            n_avg = 3
            for _ in range(n_avg):
                opt = opt_class(prob, **kwargs)
                opt.solve()
            
            avg_time = (time.time() - start) / n_avg
            times.append(avg_time)
        
        print("Done!")
        
        # Vẽ đường cho thuật toán này
        plt.plot(dims, times, marker='o', linewidth=2, label=opt_class.__name__)
        
        # Hiển thị số liệu tại điểm cuối cùng
        plt.annotate(f"{times[-1]:.4f}s", (dims[-1], times[-1]), 
                     xytext=(5, 0), textcoords="offset points", fontsize=8)

    # Trang trí biểu đồ
    plt.title(f"Scalability Comparison: Time vs Dimension")
    plt.xlabel("Problem Dimension (Size)")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend() # Hiển thị chú thích
    plt.show()