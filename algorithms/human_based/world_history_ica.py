import numpy as np
from algorithms.human_based.ica import ImperialistCompetitiveAlgorithm

class WorldHistoryICA(ImperialistCompetitiveAlgorithm):
    """
    World History ICA - Phiên bản 'Dynamic Borders' (Biên giới động).
    Thêm cơ chế rung lắc cho các Đế quốc lớn.
    """
    def __init__(self, problem, pop_size=100, n_empires=10, max_year=2026, **kwargs):
        super().__init__(problem, pop_size, n_empires, **kwargs)
        self.max_year = max_year
        self.current_year = 0
        
        # Bảng phân vai
        self.roles = {
            'ROME': None, 'MONGOL': None, 'CHINA': None, 
            'USA': None, 'USSR': None, 'EU': None, 'INDIA': None
        }

        # TIMELINE
        self.history_timeline = [
            (476, "Tây La Mã sụp đổ", "ROME_FALL"),
            (1206, "Mông Cổ trỗi dậy", "MONGOL_RISE"),
            (1279, "Mông Cổ diệt Nam Tống", "MONGOL_CONQUEST"),
            (1368, "Nhà Minh thành lập", "CHINA_RECOVERY"),
            (1492, "Tìm ra Châu Mỹ", "DISCOVERY"),
            (1945, "Trật tự Yalta (Mỹ - Liên Xô)", "COLD_WAR_START"),
            (1991, "Liên Xô tan rã -> Đa Cực", "MULTIPOLAR_WORLD"),
            (2026, "Thế giới hiện tại", "MODERN")
        ]
        self.triggered_events = set()

    def _get_current_year(self, iteration):
        progress = iteration / self.max_iter
        return int(progress * self.max_year)

    def _assign_role(self, role_name, exclude_ids=[]):
        if self.roles[role_name] is None:
            candidates = [i for i in range(self.n_empires) 
                          if i not in self.roles.values() and i not in exclude_ids]
            if candidates:
                self.roles[role_name] = np.random.choice(candidates)
            else:
                fallback = [i for i in range(self.n_empires) if i not in exclude_ids]
                if fallback: self.roles[role_name] = fallback[0]
        return self.roles[role_name]

    def _apply_event_effect(self, event_type, empire_colonies, empires_fit, empires_pos):
        """THỰC THI KỊCH BẢN LỊCH SỬ"""
        
        # --- 476: LA MÃ SỤP ĐỔ ---
        if event_type == "ROME_FALL":
            rome_idx = np.argmin(empires_fit)
            self.roles['ROME'] = rome_idx
            print(f"      -> 🏛️ Đế chế La Mã (ID {rome_idx}) sụp đổ!")
            empires_fit[rome_idx] = 1e9 
            empires_pos[rome_idx] = np.random.uniform(self.problem.bounds[:,0], self.problem.bounds[:,1])
            empire_colonies[rome_idx]['pos'] = np.empty((0, self.problem.dim))
            empire_colonies[rome_idx]['fit'] = np.array([])

        # --- 1206: MÔNG CỔ TRỖI DẬY ---
        elif event_type == "MONGOL_RISE":
            midx = self._assign_role('MONGOL')
            print(f"      -> 🐎 Mông Cổ (ID {midx}) trỗi dậy!")
            best_val = np.min(empires_fit)
            empires_fit[midx] = best_val * 0.05 

        # --- 1279: DIỆT NAM TỐNG ---
        elif event_type == "MONGOL_CONQUEST":
            midx = self.roles['MONGOL']
            cidx = self._assign_role('CHINA', exclude_ids=[midx])
            if midx is not None and cidx is not None:
                print(f"      -> ⚔️ Mông Cổ (ID {midx}) nuốt chửng Trung Hoa (ID {cidx}).")
                if len(empire_colonies[cidx]['pos']) > 0:
                    empire_colonies[midx]['pos'] = np.vstack([empire_colonies[midx]['pos'], empire_colonies[cidx]['pos']])
                    empire_colonies[midx]['fit'] = np.append(empire_colonies[midx]['fit'], empire_colonies[cidx]['fit'])
                empires_fit[cidx] = 1e9
                empire_colonies[cidx]['pos'] = np.empty((0, self.problem.dim))
                empire_colonies[cidx]['fit'] = np.array([])
                empires_pos[midx] = (empires_pos[midx] + empires_pos[cidx]) / 2

        # --- 1368: TRUNG HOA HỒI PHỤC ---
        elif event_type == "CHINA_RECOVERY":
            midx = self.roles['MONGOL']
            cidx = self.roles['CHINA']
            if midx is not None and cidx is not None:
                print(f"      -> 🐲 Nhà Minh (ID {cidx}) đánh đuổi Mông Cổ.")
                empires_pos[cidx] = empires_pos[midx] + np.random.uniform(-1, 1, self.problem.dim)
                empires_fit[cidx] = empires_fit[midx] * 0.8
                n_mongol_cols = len(empire_colonies[midx]['pos'])
                if n_mongol_cols > 0:
                    n_restore = int(n_mongol_cols * 0.6)
                    empire_colonies[cidx]['pos'] = empire_colonies[midx]['pos'][:n_restore]
                    empire_colonies[cidx]['fit'] = empire_colonies[midx]['fit'][:n_restore]
                    empire_colonies[midx]['pos'] = empire_colonies[midx]['pos'][n_restore:]
                    empire_colonies[midx]['fit'] = empire_colonies[midx]['fit'][n_restore:]
                empires_fit[midx] *= 10.0

        # --- 1945: TRẬT TỰ YALTA ---
        elif event_type == "COLD_WAR_START":
            exclude = [self.roles['CHINA']] if self.roles['CHINA'] is not None else []
            usa_idx = self._assign_role('USA', exclude_ids=exclude)
            ussr_idx = self._assign_role('USSR', exclude_ids=exclude + [usa_idx])
            
            print(f"      -> 🦅 Yalta: Mỹ (ID {usa_idx}) và Liên Xô (ID {ussr_idx}) chia đôi thế giới.")
            best_fit = np.min(empires_fit)
            empires_fit[usa_idx] = best_fit * 0.1
            empires_fit[ussr_idx] = best_fit * 0.12 
            for i in range(self.n_empires):
                if i not in [usa_idx, ussr_idx, self.roles['CHINA']]:
                    empires_fit[i] = 1e7 

        # --- 1991: ĐA CỰC ---
        elif event_type == "MULTIPOLAR_WORLD":
            ussr_idx = self.roles['USSR']
            usa_idx = self.roles['USA']
            china_idx = self.roles['CHINA']
            print(f"      -> 🌍 1992: Liên Xô tan rã. Trật tự ĐA CỰC hình thành!")
            
            free_lands_pos = np.empty((0, self.problem.dim))
            free_lands_fit = np.array([])

            # 1. Liên Xô -> Nga
            if ussr_idx is not None:
                empires_fit[ussr_idx] *= 50.0 
                cols = empire_colonies[ussr_idx]['pos']
                fits = empire_colonies[ussr_idx]['fit']
                if len(cols) > 0:
                    n_keep = int(len(cols) * 0.3)
                    empire_colonies[ussr_idx]['pos'] = cols[:n_keep]
                    empire_colonies[ussr_idx]['fit'] = fits[:n_keep]
                    free_lands_pos = cols[n_keep:]
                    free_lands_fit = fits[n_keep:]

            # 2. Kích hoạt cực mới
            exclude_list = [usa_idx, ussr_idx, china_idx]
            eu_idx = self._assign_role('EU', exclude_ids=exclude_list)
            india_idx = self._assign_role('INDIA', exclude_ids=exclude_list + [eu_idx])
            major_powers = [usa_idx, ussr_idx, china_idx, eu_idx, india_idx]
            base_power = empires_fit[usa_idx] * 5.0
            
            # Hồi sinh hàng loạt
            for i in range(self.n_empires):
                if i not in major_powers:
                    empires_pos[i] = np.random.uniform(self.problem.bounds[:,0], self.problem.bounds[:,1])
                    empires_fit[i] = base_power * np.random.uniform(2.0, 10.0)
                    if empires_fit[i] > 1e8:
                        empires_fit[i] = base_power * 10.0

            # 3. Phân chia đất trống
            if len(free_lands_pos) > 0:
                chunks = np.array_split(free_lands_pos, 3)
                chunk_fits = np.array_split(free_lands_fit, 3)
                if eu_idx is not None and len(chunks) > 0:
                    empire_colonies[eu_idx]['pos'] = np.vstack([empire_colonies[eu_idx]['pos'], chunks[0]]) if len(empire_colonies[eu_idx]['pos']) > 0 else chunks[0]
                    empire_colonies[eu_idx]['fit'] = np.append(empire_colonies[eu_idx]['fit'], chunk_fits[0])
                if china_idx is not None and len(chunks) > 1:
                    empire_colonies[china_idx]['pos'] = np.vstack([empire_colonies[china_idx]['pos'], chunks[1]]) if len(empire_colonies[china_idx]['pos']) > 0 else chunks[1]

        # --- KHÁM PHÁ ---
        elif event_type == "DISCOVERY":
            lucky_emp = np.random.randint(0, self.n_empires)
            n_explorers = int(len(empire_colonies[lucky_emp]['pos']) * 0.3)
            if n_explorers > 0:
                new_lands = np.random.uniform(self.problem.bounds[:,0], self.problem.bounds[:,1], (n_explorers, self.problem.dim))
                new_fits = np.apply_along_axis(self.problem.fitness, 1, new_lands)
                empire_colonies[lucky_emp]['pos'][:n_explorers] = new_lands
                empire_colonies[lucky_emp]['fit'][:n_explorers] = new_fits

    def _evolve(self):
        # 1. KHỞI TẠO
        dim = self.problem.dim
        lb, ub = self.problem.bounds[:, 0], self.problem.bounds[:, 1]
        pop = np.random.uniform(lb, ub, (self.pop_size, dim))
        fitness = np.apply_along_axis(self.problem.fitness, 1, pop)
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]; fitness = fitness[sorted_idx]
        self.update_global_best(pop[0], fitness[0])
        self.save_history()
        
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

        print(f"\n⏳ BẮT ĐẦU DÒNG CHẢY LỊCH SỬ (Năm 0 - {self.max_year})...\n")
        
        for t in range(self.max_iter):
            self.current_year = self._get_current_year(t)
            
            # --- CHECK EVENT ---
            for year, name, type_ in self.history_timeline:
                if self.current_year >= year and year not in self.triggered_events:
                    print(f"   >>> 📅 SỰ KIỆN: {name}")
                    self._apply_event_effect(type_, empire_colonies, empires_fit, empires)
                    self.triggered_events.add(year)

            # --- CƠ CHẾ MỚI: RUNG LẮC BIÊN GIỚI (EMPIRE DRIFT) ---
            # Mỗi năm, các Đế quốc (kể cả nước lớn) sẽ thay đổi vị trí trung tâm một chút
            # Điều này làm biên giới Voronoi liên tục thay đổi hình dạng
            drift_magnitude = 0.05 # Độ lệch nhỏ
            drift = np.random.uniform(-drift_magnitude, drift_magnitude, (self.n_empires, dim))
            
            # Chỉ áp dụng cho các nước đang sống
            living_mask = empires_fit < 1e8
            empires[living_mask] += drift[living_mask]
            empires = np.clip(empires, lb, ub)
            
            # Cập nhật lại fitness sau khi di chuyển (có thể tốt lên hoặc xấu đi)
            # Điều này tạo ra sự thăng trầm tự nhiên
            new_fits = np.apply_along_axis(self.problem.fitness, 1, empires)
            # Chỉ cập nhật fitness nếu nó không quá tệ (tránh việc vua nhảy vào chỗ chết)
            # Nhưng vẫn cho phép dao động
            empires_fit[living_mask] = new_fits[living_mask]

            # --- LOGIC ICA CHUẨN ---
            for i in range(self.n_empires):
                if empires_fit[i] > 1e8 or len(empire_colonies[i]['pos']) == 0: continue
                
                # Assimilation
                vec_diff = empires[i] - empire_colonies[i]['pos']
                move = 2.0 * np.random.rand(len(empire_colonies[i]['pos']), dim) * vec_diff
                empire_colonies[i]['pos'] += move
                empire_colonies[i]['pos'] = np.clip(empire_colonies[i]['pos'], lb, ub)
                
                # Revolution
                is_revolting = np.random.rand(len(empire_colonies[i]['pos'])) < 0.1
                if np.sum(is_revolting) > 0:
                    sigma = 0.1 * (ub - lb)
                    empire_colonies[i]['pos'][is_revolting] += np.random.normal(0, sigma, (np.sum(is_revolting), dim))
                    empire_colonies[i]['pos'] = np.clip(empire_colonies[i]['pos'], lb, ub)
                
                # Exchange
                empire_colonies[i]['fit'] = np.apply_along_axis(self.problem.fitness, 1, empire_colonies[i]['pos'])
                best_col_idx = np.argmin(empire_colonies[i]['fit'])
                if empire_colonies[i]['fit'][best_col_idx] < empires_fit[i]:
                    empires[i], empire_colonies[i]['pos'][best_col_idx] = empire_colonies[i]['pos'][best_col_idx].copy(), empires[i].copy()
                    empires_fit[i], empire_colonies[i]['fit'][best_col_idx] = empire_colonies[i]['fit'][best_col_idx], empires_fit[i]

            # Competition
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
            self.save_history()

        print(f"\n✅ KẾT THÚC DÒNG THỜI GIAN TẠI NĂM {self.current_year}.")
        return self.global_best_solution, self.global_best_fitness