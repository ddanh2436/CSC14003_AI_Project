import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_surface(problem, title="Objective Function Landscape"):
    """Vẽ địa hình 3D (Giữ nguyên như cũ)"""
    if problem.dim != 2:
        print(f"⚠️ Không thể vẽ 3D cho bài toán {problem.dim} chiều.")
        return

    print("🎨 Đang vẽ biểu đồ 3D... (Có thể mất vài giây)")
    x_min, x_max = problem.bounds[0]
    y_min, y_max = problem.bounds[1]
    
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = problem.fitness(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Fitness Value')
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.show()

# --- CẬP NHẬT MỚI: Hỗ trợ so sánh nhiều thuật toán ---
def plot_convergence(histories_dict, title="Convergence Comparison"):
    """
    Vẽ biểu đồ so sánh nhiều thuật toán trên cùng 1 hình.
    
    Args:
        histories_dict: Dictionary dạng {'Tên Thuật Toán': [list history], ...}
        title: Tiêu đề biểu đồ
    """
    plt.figure(figsize=(10, 6))
    
    # Duyệt qua từng thuật toán trong dictionary để vẽ
    for name, history in histories_dict.items():
        plt.plot(history, label=name, linewidth=2)
    
    plt.title(title, fontsize=14)
    plt.xlabel("Iterations (Vòng lặp)")
    plt.ylabel("Best Fitness (Log Scale)")
    
    # Quan trọng: Dùng thang Logarit để nhìn rõ sự khác biệt
    # Vì GA thường xuống rất thấp (10^-5) trong khi Hill Climbing kẹt ở mức cao (10^0)
    plt.yscale('log') 
    
    plt.grid(True, linestyle='--', alpha=0.7, which="both")
    plt.legend() # Hiển thị chú thích tên thuật toán
    plt.show()