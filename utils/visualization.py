import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_surface(problem, title="Objective Function Landscape"):
    """
    Vẽ địa hình hàm mục tiêu dạng 3D.
    Chỉ hoạt động với bài toán 2 chiều (dim=2) để con người có thể nhìn thấy được.
    """
    # Kiểm tra an toàn: Chỉ vẽ được 3D nếu bài toán là 2 chiều
    if problem.dim != 2:
        print(f"⚠️ Không thể vẽ 3D cho bài toán {problem.dim} chiều. Chỉ hỗ trợ 2 chiều.")
        return

    print("🎨 Đang vẽ biểu đồ 3D... (Có thể mất vài giây)")
    
    # 1. Tạo lưới điểm (Grid) để vẽ
    # Lấy giới hạn min/max từ bài toán
    x_min, x_max = problem.bounds[0]
    y_min, y_max = problem.bounds[1]
    
    # Tạo 100 điểm chia đều từ min đến max
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    
    # Tạo lưới toạ độ
    X, Y = np.meshgrid(x, y)
    
    # 2. Tính giá trị Fitness (Z) tại từng điểm trên lưới
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Gọi hàm fitness của bài toán
            Z[i, j] = problem.fitness(np.array([X[i, j], Y[i, j]]))

    # 3. Vẽ biểu đồ
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # cmap='viridis': Bảng màu từ xanh đến vàng (dễ nhìn)
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Fitness Value')
    
    # Thêm thanh màu chú thích độ cao
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.show()

def plot_convergence(history, algorithm_name="Algorithm"):
    """
    Vẽ biểu đồ đường thể hiện sự hội tụ (Fitness giảm dần theo thời gian).
    history: List chứa các giá trị fitness tốt nhất qua từng vòng lặp.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history, label=algorithm_name, color='red', linewidth=2)
    
    plt.title(f"Convergence Plot: {algorithm_name}", fontsize=14)
    plt.xlabel("Iterations (Vòng lặp)")
    plt.ylabel("Best Fitness Found")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()