import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def _norm_cdf_approx(x):
    """
    Xấp xỉ CDF của phân phối chuẩn chuẩn hóa N(0,1) bằng công thức
    Abramowitz-Stegun.
    """
    z = np.asarray(x, dtype=float)
    abs_z = np.abs(z)
    t = 1.0 / (1.0 + 0.2316419 * abs_z)
    poly = (
        0.319381530 * t
        - 0.356563782 * t**2
        + 1.781477937 * t**3
        - 1.821255978 * t**4
        + 1.330274429 * t**5
    )
    pdf = 0.3989422804014327 * np.exp(-0.5 * abs_z**2)
    cdf_pos = 1.0 - pdf * poly
    return np.where(z >= 0, cdf_pos, 1.0 - cdf_pos)

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
    if pooled_se == 0:
        return 0.0, 1.0
    t_stat = (m1 - m2) / pooled_se
    
    # Xấp xỉ P_value bằng hàm lỗi erf - Dựa trên phân phối chuẩn
    # Với n > 30, phân phối T xấp xỉ phân phối chuẩn
    cdf = _norm_cdf_approx(abs(t_stat))
    p_value = 2 * (1 - cdf)

    return float(t_stat), float(p_value)


def perform_ranksums_numpy(data1, data2):
    """
    Wilcoxon rank-sum (Mann-Whitney U) bằng NumPy thuần,
    trả về z-statistic và p-value hai phía.
    """
    sample1 = np.asarray(data1, dtype=float)
    sample2 = np.asarray(data2, dtype=float)
    n1, n2 = len(sample1), len(sample2)

    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    combined = np.concatenate([sample1, sample2])
    order = np.argsort(combined, kind='mergesort')
    sorted_values = combined[order]

    ranks_sorted = np.empty(len(combined), dtype=float)
    i = 0
    while i < len(sorted_values):
        j = i
        while j + 1 < len(sorted_values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        ranks_sorted[i:j + 1] = avg_rank
        i = j + 1

    ranks = np.empty(len(combined), dtype=float)
    ranks[order] = ranks_sorted

    r1 = np.sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0

    n = n1 + n2
    mean_u = n1 * n2 / 2.0

    if n <= 1:
        return 0.0, 1.0

    _, counts = np.unique(combined, return_counts=True)
    tie_term = np.sum(counts**3 - counts)
    var_u = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    if var_u <= 0:
        return 0.0, 1.0

    z_stat = (u1 - mean_u) / np.sqrt(var_u)
    p_value = 2 * (1 - _norm_cdf_approx(abs(z_stat)))
    return float(z_stat), float(p_value)

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