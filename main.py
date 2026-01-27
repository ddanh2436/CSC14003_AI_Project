# Import các thư viện cần thiết
from problems.continuous import Sphere, Rastrigin
from algorithms.classical.hill_climbing import HillClimbing
from utils.visualization import plot_3d_surface, plot_convergence
# LƯU Ý: Import đủ 3 hàm từ utils.metrics
from utils.metrics import run_experiment, measure_memory, run_scalability_test

def main():
    print("   CHƯƠNG TRÌNH DEMO: SEARCH & NATURE-INSPIRED ALGORITHMS")

    # --- PHẦN 1: VISUALIZATION (Vẽ 3D) ---
    print("\n>>> PHẦN 1: HIỂN THỊ ĐỊA HÌNH HÀM SỐ (3D SURFACE)")
    print("    Đang vẽ hàm Rastrigin (2 chiều)...")
    print("    (Hãy xoay cửa sổ đồ họa để xem, sau đó đóng lại để chạy tiếp)")
    
    # Chúng ta dùng hàm Rastrigin 2 chiều để vẽ 3D cho đẹp
    problem_visual = Rastrigin(dim=2) 
    plot_3d_surface(problem_visual, title="Rastrigin Function (2D Landscape)")


    # --- PHẦN 2: CHẠY THỰC NGHIỆM CƠ BẢN ---
    print("\n>>> PHẦN 2: KIỂM TRA ĐỘ HỘI TỤ & ĐỘ ỔN ĐỊNH (ROBUSTNESS)")
    
    # Tạo bài toán Sphere 10 chiều để test thuật toán
    problem_benchmark = Sphere(dim=10)
    
    # A. Chạy 1 lần demo để vẽ biểu đồ hội tụ
    print("    [2.1] Đang chạy 1 lần demo Hill Climbing để vẽ biểu đồ hội tụ...")
    hc_demo = HillClimbing(problem_benchmark, max_iter=200, step_size=0.5)
    _, _, history = hc_demo.solve()
    plot_convergence(history, algorithm_name="Hill Climbing")

    # B. Chạy thống kê 30 lần (Theo chuẩn báo cáo)
    print("    [2.2] Đang chạy thống kê 30 lần để tính Mean/Std...")
    run_experiment(
        optimizer_class=HillClimbing, 
        problem=problem_benchmark, 
        n_runs=30,       # Số lần chạy
        max_iter=500,    # Tham số: Số vòng lặp
        step_size=0.5    # Tham số: Kích thước bước nhảy
    )


    # --- PHẦN 3: KIỂM TRA NÂNG CAO (Advanced Metrics) ---
    print("\n>>> PHẦN 3: KIỂM TRA BỘ NHỚ & KHẢ NĂNG MỞ RỘNG (SCALABILITY)")

    # A. Đo bộ nhớ tiêu tốn (RAM)
    # Chúng ta test với bài toán lớn (100 chiều) để xem tốn bao nhiêu RAM
    print("    [3.1] Kiểm tra bộ nhớ tiêu thụ (Space Complexity)...")
    measure_memory(
        optimizer_class=HillClimbing, 
        problem=Sphere(dim=100), 
        max_iter=1000, 
        step_size=0.5
    )

    # B. Kiểm tra Scalability (Thời gian chạy khi size bài toán tăng dần)
    # Test các kích thước: 10, 30, 50, 100 chiều
    print("    [3.2] Kiểm tra Scalability (Vẽ biểu đồ Time vs Dimension)...")
    run_scalability_test(
        optimizer_class=HillClimbing, 
        problem_class=Sphere,  # Truyền tên Class, KHÔNG truyền object Sphere()
        dims=[10, 30, 50, 100], 
        max_iter=500, 
        step_size=0.5
    )

    print("   HOÀN TẤT TOÀN BỘ QUÁ TRÌNH KIỂM THỬ!")

if __name__ == "__main__":
    main()