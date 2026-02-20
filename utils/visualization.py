import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
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
    Vẽ biểu đồ so sánh sự hội tụ (Đã tối ưu giao diện: Đường mượt cho Swarm, Chấm tròn cho Classical).
    """
    plt.figure(figsize=(10, 6))
    
    all_values = []
    for name, history in histories_dict.items():
        # Lọc bỏ các giá trị Infinity (Vô cực)
        valid_history = [val for val in history if val != float('inf')]
        
        if valid_history:
            # --- LOGIC VẼ ĐẸP Ở ĐÂY ---
            if len(valid_history) == 1:
                # Dành cho BFS/A* (chỉ có 1 điểm): Vẽ chấm tròn to, không vẽ đường
                plt.plot(valid_history, label=name, marker='o', markersize=8, linewidth=0)
            else:
                # Dành cho Metaheuristic (nhiều điểm): Vẽ đường mượt mà, KHÔNG có chấm
                plt.plot(valid_history, label=name, linewidth=2)
                
            all_values.extend(valid_history)
            
    if not all_values:
        print(f"⚠️ Không có dữ liệu hợp lệ (toàn vô cực) để vẽ biểu đồ cho {title}")
        plt.close()
        return
        
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Iterations / Updates (Vòng lặp)")
    
    # Tự động chọn thang đo trục Y
    if min(all_values) > 0:
        plt.yscale('log')
        plt.ylabel("Best Fitness (Log Scale)")
    else:
        plt.yscale('linear')
        plt.ylabel("Best Fitness")
    
    plt.grid(True, linestyle='--', alpha=0.7, which="both")
    plt.legend()
    plt.show()
    
def plot_empire_map(empires, empire_costs, problem, iteration, title="World Order Map"):
    """
    Vẽ bản đồ lãnh thổ Đế quốc (Voronoi) CHỈ DÙNG NUMPY.
    Nguyên lý: Chia bản đồ thành lưới ô vuông nhỏ (grid), 
    mỗi ô sẽ thuộc về Đế quốc nào nằm gần nó nhất.
    """
    if problem.dim != 2:
        return

    # Tắt hiển thị plot tương tác để code chạy nhanh hơn
    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. TẠO LƯỚI ĐIỂM ẢNH (GRID) ĐỂ VẼ LÃNH THỔ
    # Độ phân giải 200x200 là đủ nét và nhanh
    resolution = 200 
    x_min, x_max = problem.bounds[0]
    y_min, y_max = problem.bounds[1]
    
    xx = np.linspace(x_min, x_max, resolution)
    yy = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(xx, yy)
    
    # Làm phẳng lưới để tính toán vector hóa (Matrix operation)
    # Shape: (N_pixels, 2)
    grid_points = np.c_[X.ravel(), Y.ravel()]
    
    # 2. TÍNH VORONOI BẰNG NUMPY (NEAREST NEIGHBOR)
    # Tính khoảng cách từ MỌI điểm grid đến MỌI đế quốc
    # Dùng Broadcasting: (N_pixels, 1, 2) - (1, N_empires, 2)
    diff = grid_points[:, np.newaxis, :] - empires[np.newaxis, :, :]
    dists = np.sum(diff**2, axis=2) # Khoảng cách bình phương
    
    # Tìm index của đế quốc gần nhất cho mỗi điểm grid
    # nearest_idx có shape (N_pixels,) chứa ID của đế quốc (0, 1, 2...)
    nearest_idx = np.argmin(dists, axis=1)
    
    # Reshape lại thành ma trận 2D để vẽ ảnh
    regions = nearest_idx.reshape(X.shape)

    # 3. VẼ LÃNH THỔ (VÙNG MÀU)
    # Dùng cmap 'tab20' để có nhiều màu phân biệt
    cmap = plt.cm.get_cmap('tab20', len(empires))
    
    # Vẽ các vùng lãnh thổ (imshow hoặc pcolormesh)
    # alpha=0.6 để màu hơi trong suốt, nhìn thấy địa hình bên dưới nếu cần
    ax.imshow(regions, extent=(x_min, x_max, y_min, y_max), 
              origin='lower', cmap=cmap, alpha=0.6, interpolation='nearest')

    # 4. TÍNH KÍCH THƯỚC BONG BÓNG (SỨC MẠNH)
    costs = np.array(empire_costs)
    worst = np.max(costs)
    best = np.min(costs)
    
    if worst == best:
        sizes = np.ones(len(costs)) * 150
    else:
        # Chuẩn hóa kích thước từ 50 đến 800
        sizes = 50 + 750 * (worst - costs) / (worst - best + 1e-9)

    # 5. VẼ VỊ TRÍ CÁC ĐẾ QUỐC (ĐIỂM TRÒN)
    # Lấy màu tương ứng từ cmap để chấm tròn trùng màu với lãnh thổ (nhưng đậm hơn)
    emp_colors = cmap(np.arange(len(empires)))
    
    ax.scatter(empires[:, 0], empires[:, 1], s=sizes, c=emp_colors, edgecolors='white', linewidth=1.5, zorder=5)
    
    # Đánh số thứ tự
    for i, (x, y) in enumerate(empires):
        ax.text(x, y, str(i), fontsize=9, ha='center', va='center', color='black', fontweight='bold', zorder=6)

    # 6. TRANG TRÍ & LƯU
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"{title}\n(Vẽ bằng NumPy Nearest Neighbor)", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Lưu file
    filename = f"history_map_{iteration:03d}.png"
    plt.savefig(filename, dpi=100) # dpi thấp chút cho nhẹ
    plt.close(fig) # Đóng figure