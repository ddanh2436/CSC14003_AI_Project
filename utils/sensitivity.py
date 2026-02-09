import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.metrics import run_experiment

def run_parameter_sweep(algo_class, problem, param_name, values, n_runs=10, **base_params):
    """
    Chạy thử nghiệm quét qua một dải giá trị của một tham số cụ thể.
    """
    results = []
    print(f"\n📊 Đang phân tích độ nhạy tham số: {param_name}")
    
    for val in values:
        current_params = base_params.copy()
        current_params[param_name] = val
        
        # Chạy thực nghiệm n_runs lần cho mỗi giá trị tham số
        stats = run_experiment(algo_class, problem, n_runs=n_runs, **current_params)
        
        # Lưu kết quả để vẽ biểu đồ
        results.append({
            'Value': val,
            'Mean Fitness': stats['mean_fitness'],
            'Std Fitness': stats['std_fitness'],
            'Best Fitness': stats['best_fitness'],
            'Avg Time': stats['avg_time']
        })
    
    return pd.DataFrame(results)

def plot_sensitivity_results(df, param_name, algo_name, problem_name):
    """
    Vẽ biểu đồ phân tích độ nhạy tham số chuyên nghiệp.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Vẽ đường Mean Fitness kèm vùng đổ bóng (Standard Deviation)
    sns.lineplot(data=df, x='Value', y='Mean Fitness', marker='o', ax=ax1, color='blue', label='Mean Fitness')
    
    # Vẽ vùng sai số (Robustness)
    ax1.fill_between(df['Value'], 
                     df['Mean Fitness'] - df['Std Fitness'], 
                     df['Mean Fitness'] + df['Std Fitness'], 
                     alpha=0.2, color='blue')

    ax1.set_xlabel(f'Tham số: {param_name}', fontsize=12)
    ax1.set_ylabel('Fitness (Thấp hơn là tốt hơn)', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Vẽ thêm trục phụ cho thời gian chạy (Computational Complexity)
    ax2 = ax1.twinx()
    sns.barplot(data=df, x='Value', y='Avg Time', ax=ax2, alpha=0.3, color='gray', label='Avg Time')
    ax2.set_ylabel('Thời gian chạy trung bình (s)', fontsize=12, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    plt.title(f'Phân tích độ nhạy {param_name}\nThuật toán: {algo_name} | Bài toán: {problem_name}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"sensitivity_{algo_name}_{param_name}.png")
    plt.show()