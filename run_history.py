import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.widgets import Button
from matplotlib.colors import ListedColormap 

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

EMPIRE_NAMES = {
    0: "Mông Cổ", 1: "Ai Cập", 2: "Trung Hoa", 3: "Hoa Kỳ", 4: "Anh Quốc", 
    5: "Pháp", 6: "Nga", 7: "Lưỡng Hà", 8: "Đức", 9: "Ba Tư", 10: "Áo", 
    11: "Ba Lan", 12: "Hy Lạp", 13: "La Mã", 14: "Ottoman", 15: "Macedonia", 
    16: "Aztec, Inca", 17: "Tây Ban Nha", 18: "Bồ Đào Nha", 19: "Mĩ La-tinh", 
    20: "Đông Nam Á", 21: "Châu Phi", 22: "Ả Rập", 23: "Châu Đại Dương"
}

CUSTOM_COLORS = [
    "#FF0000", "#D2691E", "#FFD700", "#0000FF", "#FF1493", "#00FFFF", "#008000", "#8B4513", 
    "#808080", "#800080", "#F08080", "#DC143C", "#87CEFA", "#800000", "#BDB76B", "#FFA500", 
    "#228B22", "#E6E6FA", "#000080", "#32CD32", "#20B2AA", "#A0522D", "#00FF7F", "#DDA0DD"
]

def render_fit_screen_frame(empires_pos, empire_costs, problem, year, event_name="", ax=None, fig=None):
    ax.clear()
    ax.set_aspect('auto') 
    x_min, x_max = problem.bounds[0]
    y_min, y_max = problem.bounds[1]
    
    w = x_max - x_min; h = y_max - y_min
    ax.set_xlim(x_min - w*ZOOM_OUT_LEVEL, x_max + w*ZOOM_OUT_LEVEL)
    ax.set_ylim(y_min - h*ZOOM_OUT_LEVEL, y_max + h*ZOOM_OUT_LEVEL)
    ax.set_axis_off()

    valid_mask = empire_costs < 1e8
    if not np.any(valid_mask): return

    active_empires = empires_pos[valid_mask]
    active_costs = empire_costs[valid_mask]
    active_ids = np.where(valid_mask)[0] 
    
    # Voronoi khuếch đại sức mạnh từ chi phí thực tế
    shifted_costs = active_costs - np.min(active_costs)
    powers = 1.0 / (shifted_costs + 1e-5) 
    powers = powers ** 4 
    
    xx = np.linspace(x_min, x_max, GRID_RES_X)
    yy = np.linspace(y_min, y_max, GRID_RES_Y)
    X, Y = np.meshgrid(xx, yy)
    grid_points = np.c_[X.ravel(), Y.ravel()]
    
    diff = grid_points[:, np.newaxis, :] - active_empires[np.newaxis, :, :]
    dists = np.sum(diff**2, axis=2)
    
    weighted_dists = dists / powers[np.newaxis, :]
    nearest_active_idx = np.argmin(weighted_dists, axis=1)
    
    real_ids = active_ids[nearest_active_idx]
    regions = real_ids.reshape(X.shape)

    custom_cmap = ListedColormap(CUSTOM_COLORS)
    ax.imshow(regions, extent=(x_min, x_max, y_min, y_max), 
              origin='lower', cmap=custom_cmap, alpha=0.7, interpolation='nearest', aspect='auto',
              vmin=0, vmax=23)

    unique_regions = np.unique(regions)
    for emp_idx in unique_regions:
        mask = (regions == emp_idx)
        if np.any(mask):
            center_x = np.mean(X[mask])
            center_y = np.mean(Y[mask])
            txt = ax.text(center_x, center_y, str(emp_idx), 
                          fontsize=16, ha='center', va='center', color='black', fontweight='bold', zorder=20)
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

    for txt in fig.texts: txt.set_visible(False)
    year_str = f"{abs(year)} TCN" if year < 0 else str(year)
    
    if event_name:
        fig.text(0.5, 0.96, f"NĂM {year_str} - {event_name.upper()}", 
                 ha='center', va='center', fontsize=18, color='#D32F2F', fontweight='bold')
    else:
        fig.text(0.5, 0.96, f"Năm {year_str}", 
                 ha='center', va='center', fontsize=16, color='black', fontweight='bold')

class HistoryPresenter(WorldHistoryICA):
    def __init__(self, problem, pop_size=150, n_empires=24, start_year=-3000, end_year=2026, **kwargs):
        super().__init__(problem, pop_size, n_empires, **kwargs)
        self.start_year = start_year
        self.end_year = end_year
        self.is_paused = False 
        self.btn = None 
        
        self.history_timeline = [
            (-3000, f"Bình minh văn minh: {EMPIRE_NAMES[1]} (1) & {EMPIRE_NAMES[7]} (7)", "ANCIENT_START"),
            (-1500, f"{EMPIRE_NAMES[2]} (2) hình thành bên dòng Hoàng Hà", "CHINA_START"),
            (-330,  f"Alexander Đại Đế: {EMPIRE_NAMES[15]} (15) chinh phạt {EMPIRE_NAMES[9]} (9)", "MACEDONIA_RISE"),
            (-27,   f"{EMPIRE_NAMES[13]} (13) thống nhất Địa Trung Hải", "ROME_RISE"),
            (476,   f"{EMPIRE_NAMES[13]} (13) sụp đổ, Châu Âu chìm trong Đêm Trường", "ROME_FALL"),
            (632,   f"Sự trỗi dậy của Hồi giáo và {EMPIRE_NAMES[22]} (22)", "ARAB_RISE"),
            (1206,  f"Đế quốc {EMPIRE_NAMES[0]} (0) trỗi dậy", "MONGOL_RISE"),
            (1279,  f"{EMPIRE_NAMES[0]} (0) chinh phạt Á-Âu, diệt {EMPIRE_NAMES[2]} (2)", "MONGOL_CONQUEST"),
            (1368,  f"Nhà Minh {EMPIRE_NAMES[2]} (2) đánh đuổi {EMPIRE_NAMES[0]} (0)", "CHINA_RECOVERY"),
            (1453,  f"{EMPIRE_NAMES[14]} (14) chiếm Constantinople", "OTTOMAN_RISE"),
            (1492,  f"Khám phá Châu Mỹ: {EMPIRE_NAMES[17]} (17), {EMPIRE_NAMES[18]} (18)", "DISCOVERY"),
            (1776,  f"{EMPIRE_NAMES[3]} (3) giành độc lập", "US_INDEPENDENCE"),
            (1815,  f"{EMPIRE_NAMES[4]} (4) thành Đế quốc mặt trời không lặn", "BRITISH_EMPIRE"),
            (1914,  f"Thế chiến 1: {EMPIRE_NAMES[8]} (8), {EMPIRE_NAMES[14]} (14) vs {EMPIRE_NAMES[4]} (4), {EMPIRE_NAMES[5]} (5)", "WW1"),
            (1939,  f"Thế chiến 2: Trục Phát xít bành trướng", "WW2"),
            (1945,  f"Trật tự Yalta: {EMPIRE_NAMES[3]} (3) và {EMPIRE_NAMES[6]} (6) chia đôi thế giới", "COLD_WAR"),
            (1991,  f"{EMPIRE_NAMES[6]} (6) tan rã, Kỷ nguyên đơn cực {EMPIRE_NAMES[3]} (3)", "SOVIET_COLLAPSE"),
            (2010,  f"{EMPIRE_NAMES[2]} (2) trỗi dậy thành siêu cường kinh tế", "CHINA_MODERN"),
            (2026,  f"Thế giới hiện tại - KẾT THÚC", "MODERN_2026")
        ]
        
    def _get_current_year(self, iteration):
        progress = iteration / (self.max_iter - 1)
        return int(self.start_year + progress * (self.end_year - self.start_year))

    def toggle_pause(self, event=None):
        self.is_paused = not self.is_paused
        if self.btn:
            self.btn.label.set_text('▶' if self.is_paused else '⏸')
            self.btn.ax.figure.canvas.draw_idle()
            
    def on_key_press(self, event):
        if event.key == ' ': self.toggle_pause()

    def _apply_event_effect(self, event_type, empire_colonies, empires_fit, empires_pos):
        best_fit = np.min(empires_fit)
        lb, ub = self.problem.bounds[:, 0], self.problem.bounds[:, 1]
        
        # HÀM HỒI SINH HOẶC BUFF SỨC MẠNH VÀO LÕI THUẬT TOÁN
        def buff_nation(idx, multiplier=0.01):
            if empires_fit[idx] > 1e8 or len(empire_colonies[idx]['pos']) == 0:
                empires_fit[idx] = best_fit * 10 
                empires_pos[idx] = np.random.uniform(lb, ub)
                # Cấp 1 thuộc địa để có thể sinh tồn trong thuật toán ICA
                col_pos = np.random.uniform(lb, ub, (1, self.problem.dim))
                col_fit = np.apply_along_axis(self.problem.fitness, 1, col_pos)
                empire_colonies[idx]['pos'] = col_pos
                empire_colonies[idx]['fit'] = col_fit
            empires_fit[idx] = np.min(empires_fit) * multiplier
            
        # HÀM DIỆT VONG: Xóa sạch thuộc địa khỏi bộ nhớ
        def destroy_nation(idx):
            empires_fit[idx] = 1e9
            empire_colonies[idx]['pos'] = np.empty((0, self.problem.dim))
            empire_colonies[idx]['fit'] = np.array([])
            
        # HÀM THÔN TÍNH: Lấy toàn bộ mảng thuộc địa của kẻ thù nạp vào mình
        def absorb(conqueror, victim):
            if len(empire_colonies[victim]['pos']) > 0:
                empire_colonies[conqueror]['pos'] = np.vstack([empire_colonies[conqueror]['pos'], empire_colonies[victim]['pos']])
                empire_colonies[conqueror]['fit'] = np.append(empire_colonies[conqueror]['fit'], empire_colonies[victim]['fit'])
            # Di chuyển quân chủ lực tới giữa lãnh thổ kẻ thù
            empires_pos[conqueror] = (empires_pos[conqueror] + empires_pos[victim]) / 2.0
            destroy_nation(victim)

        # THỰC THI SỰ KIỆN TRÊN BỘ NHỚ THUẬT TOÁN
        if event_type == "ANCIENT_START":
            for i in range(self.n_empires): destroy_nation(i) 
            buff_nation(1, 0.1) 
            buff_nation(7, 0.1) 
        elif event_type == "CHINA_START": buff_nation(2, 0.001)
        elif event_type == "MACEDONIA_RISE":
            buff_nation(15, 0.005)
            absorb(15, 9); absorb(15, 12) 
        elif event_type == "ROME_RISE":
            buff_nation(13, 0.001)
            absorb(13, 1); absorb(13, 15)
        elif event_type == "ROME_FALL":
            destroy_nation(13)
            buff_nation(5, 5.0); buff_nation(8, 5.0); buff_nation(4, 5.0)
        elif event_type == "ARAB_RISE":
            buff_nation(22, 0.005); absorb(22, 9) 
        elif event_type == "MONGOL_RISE": buff_nation(0, 0.0001)
        elif event_type == "MONGOL_CONQUEST":
            absorb(0, 2); absorb(0, 7) # Thực sự nuốt chửng thuộc địa của Trung Hoa
            buff_nation(0, 0.000001) # Chi phí cực nhỏ -> quyền lực vô cực
            empires_fit[6] = 1e8 
        elif event_type == "CHINA_RECOVERY":
            buff_nation(2, 0.001); destroy_nation(0) # Nhà Nguyên (Mông Cổ) sụp đổ hoàn toàn
        elif event_type == "OTTOMAN_RISE": buff_nation(14, 0.05)
        elif event_type == "DISCOVERY":
            buff_nation(17, 0.01); buff_nation(18, 0.01); destroy_nation(16)    
        elif event_type == "US_INDEPENDENCE": buff_nation(3, 0.5)
        elif event_type == "BRITISH_EMPIRE":
            buff_nation(4, 0.0001); empires_fit[17] *= 50; empires_fit[5] *= 10  
        elif event_type == "WW1":
            buff_nation(8, 0.001); destroy_nation(14); destroy_nation(10); buff_nation(11, 2.0)   
        elif event_type == "WW2":
            buff_nation(8, 0.00001); destroy_nation(5)  # Đức Quốc Xã bá chủ
        elif event_type == "COLD_WAR":
            destroy_nation(8) 
            
            # YALTA: THAO TÁC TRỰC TIẾP LÊN VỊ TRÍ VÀ SỨC MẠNH LÕI
            buff_nation(3, 1e-10) # Hoa Kỳ quyền lực vô cực
            buff_nation(6, 1e-10) # Liên Xô quyền lực vô cực
            
            empires_pos[3] = lb + 0.15 * (ub - lb) # Định vị cứng Hoa Kỳ trong lõi
            empires_pos[6] = ub - 0.15 * (ub - lb) # Định vị cứng Liên Xô trong lõi
            
            # Suy yếu toàn bộ các nước còn lại
            for i in range(self.n_empires):
                if i not in [3, 6] and empires_fit[i] < 1e8: 
                    empires_fit[i] *= 1e6 
                        
        elif event_type == "SOVIET_COLLAPSE":
            destroy_nation(6) # Liên Xô tan rã
            buff_nation(6, 1000) # Nga hồi sinh nhưng rất yếu
            buff_nation(3, 1e-10) # Kỷ nguyên đơn cực
        elif event_type == "CHINA_MODERN": buff_nation(2, 0.00001) 
        elif event_type == "MODERN_2026":
            buff_nation(3, 0.001); buff_nation(2, 0.001)
            buff_nation(6, 0.005); buff_nation(20, 0.01); buff_nation(8, 0.01); buff_nation(4, 0.01)

    def _evolve(self):
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

        print(f"\n🎥 SẴN SÀNG TRÌNH CHIẾU DÒNG CHẢY LỊCH SỬ THẾ GIỚI!")
        
        plt.ion() 
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_axes([0, 0, 1, 0.90])
        
        ax_btn = fig.add_axes([0.01, 0.94, 0.035, 0.04]) 
        self.btn = Button(ax_btn, '⏸', color='black', hovercolor='gray')
        self.btn.label.set_color('white')
        self.btn.label.set_fontsize(18) 
        self.btn.on_clicked(self.toggle_pause)

        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        try:
            manager = plt.get_current_fig_manager()
            if hasattr(manager, 'window') and hasattr(manager.window, 'state'):
                manager.window.state('zoomed')
            else: manager.full_screen_toggle()
        except: pass

        for t in range(self.max_iter):
            while self.is_paused: plt.pause(0.1)

            logic_year = self._get_current_year(t)
            is_event = False; event_txt = ""; display_year = logic_year

            for year, name, type_ in self.history_timeline:
                if logic_year >= year and year not in self.triggered_events:
                    print(f"\n   >>> 📅 SỰ KIỆN: {name} (Năm {year})")
                    self._apply_event_effect(type_, empire_colonies, empires_fit, empires)
                    self.triggered_events.add(year)
                    is_event = True
                    event_txt = name
                    display_year = year 
                    break 
            
            should_draw = is_event
            
            if should_draw:
                # Đã loại bỏ khối VISUAL OVERRIDE.
                # Truyền trực tiếp dữ liệu lõi của thuật toán ra màn hình
                render_fit_screen_frame(empires, empires_fit, self.problem, display_year, event_txt, ax=ax, fig=fig)
                plt.draw(); plt.pause(DELAY_TIME)
                
                filename = f"{OUTPUT_DIR}/frame_{t:04d}_Year_{display_year}.svg"
                plt.savefig(filename, format='svg', bbox_inches='tight')

            # --- LÕI ICA CHẠY TIẾP DỰA TRÊN STATE MỚI ---
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

        print(f"\n✅ ĐÃ XONG! Mời bạn xem kết quả.")
        plt.ioff(); plt.show()
        return self.global_best_solution, self.global_best_fitness

def main():
    problem = Rastrigin(dim=2)
    algo = HistoryPresenter(
        problem, 
        pop_size=300,    
        n_empires=24, 
        start_year=-3000,
        end_year=2026,
        max_iter=800     
    )
    best_sol, best_fit = algo.solve()

if __name__ == "__main__":
    main()