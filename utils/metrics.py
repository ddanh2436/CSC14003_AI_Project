import numpy as np
import time
import tracemalloc  # Thư viện chuẩn của Python để đo bộ nhớ (RAM)
import matplotlib.pyplot as plt # Để vẽ biểu đồ Scalability

def run_experiment(optimizer_class, problem, n_runs=30, **kwargs):
    """
    Chạy thuật toán n_runs lần để lấy số liệu thống kê (Robustness).
    
    Args:
        optimizer_class: Tên Class thuật toán (VD: HillClimbing) - KHÔNG PHẢI instance
        problem: Bài toán cần giải (Object đã khởi tạo)
        n_runs: Số lần chạy thử nghiệm (Đồ án yêu cầu 30)
        **kwargs: Các tham số của thuật toán (max_iter, step_size...)
        
    Returns:
        dict: Chứa các chỉ số thống kê (mean, std, best, worst, avg_time)
    """
    fitness_results = []
    time_results = []
    
    print(f"\n📊 Đang chạy thực nghiệm {n_runs} lần cho {optimizer_class.__name__}...")
    
    for i in range(n_runs):
        # 1. Khởi tạo thuật toán mới hoàn toàn
        optimizer = optimizer_class(problem, **kwargs)
        
        # 2. Chạy giải
        _, best_fitness, _ = optimizer.solve()
        
        # 3. Lưu kết quả
        fitness_results.append(best_fitness)
        time_results.append(optimizer.run_time)
        
        # In dấu chấm để biết chương trình đang chạy
        if (i+1) % 5 == 0:
            print(f"   Run {i+1}/{n_runs} complete...", end="\r")

    print("\n   ✅ Hoàn tất thực nghiệm!")
    
    # 4. Tính toán thống kê
    stats = {
        "algorithm": optimizer_class.__name__,
        "problem": problem.name,
        "mean_fitness": np.mean(fitness_results),
        "std_fitness": np.std(fitness_results), # Độ lệch chuẩn (Robustness)
        "best_fitness": np.min(fitness_results),
        "worst_fitness": np.max(fitness_results),
        "avg_time": np.mean(time_results)
    }
    
    # 5. In báo cáo
    print("-" * 50)
    print(f"REPORT: {stats['algorithm']} on {stats['problem']}")
    print("-" * 50)
    print(f"Runs          : {n_runs}")
    print(f"Fitness (Mean): {stats['mean_fitness']:.6f}")
    print(f"Fitness (Std) : ± {stats['std_fitness']:.6f} (Độ ổn định)")
    print(f"Best Found    : {stats['best_fitness']:.6f}")
    print(f"Avg Time      : {stats['avg_time']:.4f} seconds")
    print("-" * 50)
    
    return stats

def measure_memory(optimizer_class, problem, **kwargs):
    """
    Đo lượng RAM tiêu tốn (Space Complexity).
    Sử dụng thư viện tracemalloc để theo dõi cấp phát bộ nhớ.
    """
    print(f"💾 Đang đo bộ nhớ cho {optimizer_class.__name__}...", end="")
    
    tracemalloc.start() # Bắt đầu theo dõi
    
    # Chạy thuật toán 1 lần
    opt = optimizer_class(problem, **kwargs)
    opt.solve()
    
    # Lấy thông số bộ nhớ: current (hiện tại), peak (đỉnh điểm)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop() # Dừng theo dõi
    
    # Đổi từ Byte sang MB
    peak_mb = peak / (1024 * 1024)
    print(f" Done! Peak Memory: {peak_mb:.4f} MB")
    
    return peak_mb

def run_scalability_test(optimizer_class, problem_class, dims=[10, 30, 50, 100], **kwargs):
    """
    Test khả năng mở rộng (Scalability): 
    Chạy thuật toán với kích thước bài toán (dimension) tăng dần.
    
    Args:
        optimizer_class: Class thuật toán (VD: HillClimbing)
        problem_class: Class bài toán (VD: Sphere) - Lưu ý truyền Class, không phải Object
        dims: Danh sách các số chiều cần test
        **kwargs: Tham số thuật toán
    """
    times = []
    fitnesses = []
    
    print(f"\n📈 Đang chạy kiểm tra Scalability (Mở rộng) cho {optimizer_class.__name__}...")
    
    for d in dims:
        print(f"   Testing dimension: {d}...", end="\r")
        
        # Tạo bài toán mới với số chiều d
        prob = problem_class(dim=d)
        
        # Chạy thực nghiệm (chạy 5 lần mỗi mức để lấy trung bình thời gian)
        # Tắt in log chi tiết trong run_experiment để đỡ rối màn hình
        stats = run_experiment(optimizer_class, prob, n_runs=5, **kwargs)
        
        times.append(stats['avg_time'])
        fitnesses.append(stats['mean_fitness'])
    
    print(f"\n   ✅ Hoàn tất Scalability Test trên các chiều: {dims}")

    # --- Vẽ biểu đồ Time Scalability ---
    plt.figure(figsize=(10, 6))
    plt.plot(dims, times, marker='o', linestyle='-', color='purple', linewidth=2)
    
    plt.title(f"Scalability Analysis: {optimizer_class.__name__} on {problem_class.__name__}")
    plt.xlabel("Problem Dimension (Size)")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Hiển thị giá trị cụ thể lên từng điểm
    for i, txt in enumerate(times):
        # Nếu thời gian < 0.01 giây thì hiển thị 5 số lẻ, ngược lại hiển thị 2 số lẻ
        if txt < 0.01:
            label = f"{txt:.5f}s"
        else:
            label = f"{txt:.2f}s"
            
        plt.annotate(
            label, 
            (dims[i], times[i]), 
            textcoords="offset points", 
            xytext=(0,10), 
            ha='center',
            fontsize=9,
            color='blue'
        )

    plt.show()