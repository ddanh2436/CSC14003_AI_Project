import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.widgets import Button # <--- Import thêm Widget Button

# Import các thành phần từ project
from problems.continuous import Rastrigin 
from algorithms.human_based.world_history_ica import WorldHistoryICA

# --- CẤU HÌNH ---
DELAY_TIME = 2.0    
GRID_RES_X = 800    
GRID_RES_Y = 500
ZOOM_OUT_LEVEL = 0.1  
OUTPUT_DIR = "history_svg_frames" 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def render_fit_screen_frame(empires, empire_costs, problem, year, event_name="", ax=None, fig=None):
    ax.clear()
    
    # 1. CẤU HÌNH CAMERA
    ax.set_aspect('auto') 
    x_min, x_max = problem.bounds[0]
    y_min, y_max = problem.bounds[1]
    
    # Zoom Out
    w = x_max - x_min
    h = y_max - y_min
    pad_x = w * ZOOM_OUT_LEVEL
    pad_y = h * ZOOM_OUT_LEVEL
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_axis_off()

    # --- BƯỚC LỌC ---
    valid_mask = empire_costs < 1e8
    if not np.any(valid_mask): return

    active_empires = empires[valid_mask]
    active_ids = np.where(valid_mask)[0] 
    
    # 2. VẼ NỀN VORONOI
    xx = np.linspace(x_min, x_max, GRID_RES_X)
    yy = np.linspace(y_min, y_max, GRID_RES_Y)
    X, Y = np.meshgrid(xx, yy)
    grid_points = np.c_[X.ravel(), Y.ravel()]
    
    diff = grid_points[:, np.newaxis, :] - active_empires[np.newaxis, :, :]
    dists = np.sum(diff**2, axis=2)
    nearest_active_idx = np.argmin(dists, axis=1)
    real_ids = active_ids[nearest_active_idx]
    regions = real_ids.reshape(X.shape)

    cmap = plt.get_cmap('tab20')
    ax.imshow(regions, extent=(x_min, x_max, y_min, y_max), 
              origin='lower', cmap=cmap, alpha=0.7, interpolation='nearest', aspect='auto')

    # 3. VẼ SỐ HIỆU
    unique_regions = np.unique(regions)
    for emp_idx in unique_regions:
        mask = (regions == emp_idx)
        if np.any(mask):
            center_x = np.mean(X[mask])
            center_y = np.mean(Y[mask])
            txt = ax.text(center_x, center_y, str(emp_idx), 
                          fontsize=16, ha='center', va='center', 
                          color='black', fontweight='bold', zorder=20)
            txt.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='white')])

    # 4. TIÊU ĐỀ
    # Xóa text cũ trừ text của Button (Button text nằm trong Axes khác nên ko lo)
    for txt in fig.texts:
        txt.set_visible(False)

    if event_name:
        fig.text(0.5, 0.95, f"NĂM {year} - {event_name.upper()}", 
                 ha='center', va='center',
                 fontsize=28, color='#D32F2F', fontweight='bold')
    else:
        fig.text(0.5, 0.95, f"Năm {year}", 
                 ha='center', va='center',
                 fontsize=24, color='black', fontweight='bold')

class HistoryPresenter(WorldHistoryICA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_paused = False
        self.btn_pause = None # Lưu trữ đối tượng Button

    def toggle_pause(self, event):
        """Hàm xử lý khi nhấn nút hoặc phím Space"""
        self.is_paused = not self.is_paused
        
        # Cập nhật giao diện nút bấm
        if self.btn_pause:
            if self.is_paused:
                self.btn_pause.label.set_text("▶ TIẾP TỤC")
                self.btn_pause.color = '#4CAF50' # Xanh lá
                self.btn_pause.hovercolor = '#45a049'
            else:
                self.btn_pause.label.set_text("⏸ TẠM DỪNG")
                self.btn_pause.color = '#f0f0f0' # Trắng xám
                self.btn_pause.hovercolor = '#e0e0e0'
        
        plt.draw() # Vẽ lại nút ngay lập tức

    def on_key_press(self, event):
        if event.key == ' ':
            self.toggle_pause(None)

    def _evolve(self):
        # KHỞI TẠO (Giữ nguyên)
        dim = self.problem.dim
        lb, ub = self.problem.bounds[:, 0], self.problem.bounds[:, 1]
        pop = np.random.uniform(lb, ub, (self.pop_size, dim))
        fitness = np.apply_along_axis(self.problem.fitness, 1, pop)
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]; fitness = fitness[sorted_idx]
        self.update_global_best(pop[0], fitness[0])
        
        empires = pop[:self.n_empires].copy()
        empires_fit = fitness[:self.n_empires].copy()
        colonies = pop[self.n_empires:].copy()
        colonies_fit = fitness[self.n_empires:].copy()
        
        normalized_power = (np.max(empires_fit) - empires_fit) + 0.1
        prob_power = normalized_power / np.sum(normalized_power)
        n_cols_per_empire = np.round(prob_power * (self.pop_size - self.n_empires)).astype(int)
        n_cols_per_empire[-1] += (self.pop_size - self.n_empires) - np.sum(n_cols_per_empire)
        empire_colonies = []
        start_idx = 0
        for n in n_cols_per_empire:
            end_idx = start_idx + n
            empire_colonies.append({'pos': colonies[start_idx:end_idx], 'fit': colonies_fit[start_idx:end_idx]})
            start_idx = end_idx

        print("\n🎥 ĐANG CHIẾU (CÓ NÚT TẠM DỪNG)...")
        plt.ion() 
        
        fig = plt.figure(figsize=(16, 9))
        
        # Layout: Dành 90% dưới cho bản đồ
        ax = fig.add_axes([0, 0, 1, 0.90])
        
        # --- TẠO NÚT BẤM (BUTTON) ---
        # Vị trí: [Left, Bottom, Width, Height] (Tính theo tỉ lệ 0-1 của cửa sổ)
        # Góc trên bên trái, nằm trong vùng Header trắng
        ax_btn = fig.add_axes([0.01, 0.92, 0.08, 0.05]) 
        self.btn_pause = Button(ax_btn, '⏸ TẠM DỪNG', color='#f0f0f0', hovercolor='#e0e0e0')
        self.btn_pause.label.set_fontsize(10)
        self.btn_pause.label.set_fontweight('bold')
        self.btn_pause.on_clicked(self.toggle_pause)
        
        # Vẫn giữ phím Space cho tiện
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        try:
            manager = plt.get_current_fig_manager()
            if hasattr(manager, 'window') and hasattr(manager.window, 'state'):
                manager.window.state('zoomed')
            else:
                manager.full_screen_toggle()
        except:
            pass

        for t in range(self.max_iter):
            # --- LOGIC TẠM DỪNG ---
            # Nếu đang pause, vòng lặp sẽ kẹt ở đây nhưng vẫn cho phép tương tác GUI
            while self.is_paused:
                plt.pause(0.1) 

            # --- TIẾP TỤC ---
            self.current_year = self._get_current_year(t)
            is_event = False
            event_txt = ""

            for year, name, type_ in self.history_timeline:
                if self.current_year >= year and year not in self.triggered_events:
                    print(f"   >>> 📅 SỰ KIỆN: {name} (Năm {year})")
                    self._apply_event_effect(type_, empire_colonies, empires_fit, empires)
                    self.triggered_events.add(year)
                    is_event = True
                    event_txt = name
            
            should_draw = is_event or (t % 15 == 0)
            
            if should_draw:
                render_fit_screen_frame(empires, empires_fit, self.problem, self.current_year, event_txt, ax=ax, fig=fig)
                plt.draw()
                plt.pause(DELAY_TIME)
                
                filename = f"{OUTPUT_DIR}/frame_{t:04d}_Year_{self.current_year}.svg"
                plt.savefig(filename, format='svg', bbox_inches='tight')

            # Logic ICA
            for i in range(self.n_empires):
                if empires_fit[i] > 1e8 or len(empire_colonies[i]['pos']) == 0: continue
                vec_diff = empires[i] - empire_colonies[i]['pos']
                move = 2.0 * np.random.rand(len(empire_colonies[i]['pos']), dim) * vec_diff
                empire_colonies[i]['pos'] += move
                empire_colonies[i]['pos'] = np.clip(empire_colonies[i]['pos'], lb, ub)
                is_revolting = np.random.rand(len(empire_colonies[i]['pos'])) < 0.1
                if np.sum(is_revolting) > 0:
                    sigma = 0.1 * (ub - lb)
                    empire_colonies[i]['pos'][is_revolting] += np.random.normal(0, sigma, (np.sum(is_revolting), dim))
                    empire_colonies[i]['pos'] = np.clip(empire_colonies[i]['pos'], lb, ub)
                empire_colonies[i]['fit'] = np.apply_along_axis(self.problem.fitness, 1, empire_colonies[i]['pos'])
                best_col_idx = np.argmin(empire_colonies[i]['fit'])
                if empire_colonies[i]['fit'][best_col_idx] < empires_fit[i]:
                    empires[i], empire_colonies[i]['pos'][best_col_idx] = empire_colonies[i]['pos'][best_col_idx].copy(), empires[i].copy()
                    empires_fit[i], empire_colonies[i]['fit'][best_col_idx] = empire_colonies[i]['fit'][best_col_idx], empires_fit[i]

            total_costs = np.zeros(self.n_empires)
            for i in range(self.n_empires):
                if empires_fit[i] > 1e8: total_costs[i] = -1 
                else:
                    mean_fit = np.mean(empire_colonies[i]['fit']) if len(empire_colonies[i]['fit']) > 0 else 0
                    total_costs[i] = empires_fit[i] + 0.1 * mean_fit
            
            active_indices = np.where(total_costs > -0.5)[0]
            if len(active_indices) > 1:
                sub_costs = total_costs[active_indices]
                weakest_local_idx = np.argmax(sub_costs)
                weakest_emp_idx = active_indices[weakest_local_idx]
                
                if len(empire_colonies[weakest_emp_idx]['pos']) > 0:
                    weakest_col_idx = np.argmax(empire_colonies[weakest_emp_idx]['fit'])
                    col_pos = empire_colonies[weakest_emp_idx]['pos'][weakest_col_idx].copy()
                    col_fit = empire_colonies[weakest_emp_idx]['fit'][weakest_col_idx]
                    empire_colonies[weakest_emp_idx]['pos'] = np.delete(empire_colonies[weakest_emp_idx]['pos'], weakest_col_idx, axis=0)
                    empire_colonies[weakest_emp_idx]['fit'] = np.delete(empire_colonies[weakest_emp_idx]['fit'], weakest_col_idx)
                    comp_powers = np.max(sub_costs) - sub_costs
                    if np.sum(comp_powers) > 0:
                        probs = comp_powers / np.sum(comp_powers)
                        winning_local_idx = np.random.choice(range(len(active_indices)), p=probs)
                        winning_idx = active_indices[winning_local_idx]
                        empire_colonies[winning_idx]['pos'] = np.vstack([empire_colonies[winning_idx]['pos'], col_pos])
                        empire_colonies[winning_idx]['fit'] = np.append(empire_colonies[winning_idx]['fit'], col_fit)

            min_emp_idx = np.argmin(empires_fit)
            if empires_fit[min_emp_idx] < self.global_best_fitness:
                self.update_global_best(empires[min_emp_idx], empires_fit[min_emp_idx])

        render_fit_screen_frame(empires, empires_fit, self.problem, self.max_year, "KẾT THÚC", ax=ax, fig=fig)
        filename = f"{OUTPUT_DIR}/FINAL_RESULT.svg"
        plt.savefig(filename, format='svg', bbox_inches='tight')
        
        print(f"\n✅ ĐÃ XONG! Mời bạn xem kết quả.")
        plt.ioff()
        plt.show()
        
        return self.global_best_solution, self.global_best_fitness

def main():
    problem = Rastrigin(dim=2)
    algo = HistoryPresenter(
        problem, 
        pop_size=250,    
        n_empires=50,    
        max_year=2026, 
        max_iter=600     
    )
    best_sol, best_fit = algo.solve()

if __name__ == "__main__":
    main()