import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def perform_ttest_numpy(data1, data2):
    """
    Thực hiện kiểm định t-test độc lập (Independent T-test) bằng NumPy thuần.
    Đáp ứng yêu cầu: 'Use knowledge in Statistic course to perform hypothesis testing'.
    """
    n1, n2 = len(data1), len(data2)
    m1, m2 = np.mean(data1), np.mean(data2)
    v1, v2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
    
    # Tính t-statistic
    pooled_se = np.sqrt(v1/n1 + v2/n2)
    t_stat = (m1 - m2) / pooled_se
    
    # Báo cáo: Nếu |t| > 2 (xấp xỉ mức alpha=0.05), sự khác biệt có ý nghĩa
    return t_stat, "Significant" if abs(t_stat) > 2.1 else "Not Significant"

def plot_exploration_exploitation(diversity_hist, fitness_hist, title="Exploration vs Exploitation"):
    """
    Vẽ biểu đồ phân tích hành vi Exploration-Exploitation[cite: 85].
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Trục 1: Độ đa dạng (Exploration)
    ax1.plot(diversity_hist, color='green', label='Population Diversity (Exploration)')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Diversity Index', color='green')
    
    # Trục 2: Best Fitness (Exploitation)
    ax2 = ax1.twinx()
    ax2.plot(fitness_hist, color='red', linestyle='--', label='Best Fitness (Exploitation)')
    ax2.set_ylabel('Best Fitness', color='red')
    ax2.set_yscale('log')
    
    plt.title(title)
    fig.tight_layout()
    plt.show()

def run_advanced_sensitivity(algo_class, problem, param_name, values, n_runs=15, **base_params):
    """
    Phân tích độ nhạy kèm theo kiểm định thống kê và bảng so sánh.
    """
    all_results = []
    config_data = {} # Lưu raw data để chạy t-test
    
    print(f"\n--- Advanced Sensitivity Analysis for {param_name} ---")
    
    for val in values:
        current_params = base_params.copy()
        current_params[param_name] = val
        
        runs_fitness = []
        for _ in range(n_runs):
            opt = algo_class(problem, **current_params)
            _, best_fit, _ = opt.solve()
            runs_fitness.append(best_fit)
        
        config_data[val] = runs_fitness
        all_results.append({
            'Value': val,
            'Mean': np.mean(runs_fitness),
            'Std': np.std(runs_fitness),
            'Best': np.min(runs_fitness)
        })

    # So sánh cặp tốt nhất và tệ nhất bằng T-test
    vals = list(config_data.keys())
    t_val, sig = perform_ttest_numpy(config_data[vals[0]], config_data[vals[-1]])
    print(f"📊 Statistical Comparison ({vals[0]} vs {vals[-1]}): t={t_val:.2f}, {sig}")
    
    return pd.DataFrame(all_results)