import numpy as np
from algorithms.optimizer import Optimizer

class ImperialistCompetitiveAlgorithm(Optimizer):
    """
    Imperialist Competitive Algorithm (ICA) - Base Class.
    Phiên bản gốc tối ưu hóa bằng NumPy Vectorization.
    """
    def __init__(self, problem, pop_size=100, n_empires=10, assimilation_coeff=2, revolution_prob=0.1, **kwargs):
        super().__init__(problem, pop_size=pop_size, **kwargs)
        self.n_empires = n_empires
        self.n_colonies = pop_size - n_empires
        self.beta = assimilation_coeff
        self.p_revolution = revolution_prob

    def _evolve(self):
        # 1. KHỞI TẠO
        dim = self.problem.dim
        lb, ub = self.problem.bounds[:, 0], self.problem.bounds[:, 1]
        
        pop = np.random.uniform(lb, ub, (self.pop_size, dim))
        fitness = np.apply_along_axis(self.problem.fitness, 1, pop)
        
        # Sắp xếp để chọn ra các Đế quốc mạnh nhất
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]
        
        self.update_global_best(pop[0], fitness[0])
        self.save_history()

        # Phân chia Đế quốc và Thuộc địa
        empires = pop[:self.n_empires].copy()
        empires_fit = fitness[:self.n_empires].copy()
        
        colonies = pop[self.n_empires:].copy()
        colonies_fit = fitness[self.n_empires:].copy()
        
        # Tính toán sức mạnh để chia thuộc địa
        normalized_power = (np.max(empires_fit) - empires_fit) + 0.1
        prob_power = normalized_power / np.sum(normalized_power)
        
        n_cols_per_empire = np.round(prob_power * self.n_colonies).astype(int)
        # Sửa lỗi làm tròn
        diff = self.n_colonies - np.sum(n_cols_per_empire)
        n_cols_per_empire[-1] += diff
        
        # Danh sách quản lý thuộc địa của từng đế quốc
        empire_colonies = []
        start_idx = 0
        for n in n_cols_per_empire:
            end_idx = start_idx + n
            empire_colonies.append({
                'pos': colonies[start_idx:end_idx], 
                'fit': colonies_fit[start_idx:end_idx]
            })
            start_idx = end_idx

        # 2. VÒNG LẶP CHÍNH
        for _ in range(self.max_iter):
            for i in range(self.n_empires):
                # Nếu đế quốc đã mất hết thuộc địa thì bỏ qua
                if len(empire_colonies[i]['pos']) == 0:
                    continue
                
                # A. ASSIMILATION (Đồng hóa)
                # Thuộc địa di chuyển về phía Đế quốc
                vec_diff = empires[i] - empire_colonies[i]['pos']
                # Vector hóa: nhân random cho từng dòng
                rand_vec = np.random.rand(len(empire_colonies[i]['pos']), dim)
                move = self.beta * rand_vec * vec_diff
                
                empire_colonies[i]['pos'] += move
                empire_colonies[i]['pos'] = np.clip(empire_colonies[i]['pos'], lb, ub)
                
                # B. REVOLUTION (Cách mạng)
                is_revolting = np.random.rand(len(empire_colonies[i]['pos'])) < self.p_revolution
                if np.sum(is_revolting) > 0:
                    sigma = 0.1 * (ub - lb)
                    noise = np.random.normal(0, sigma, (np.sum(is_revolting), dim))
                    empire_colonies[i]['pos'][is_revolting] += noise
                    empire_colonies[i]['pos'] = np.clip(empire_colonies[i]['pos'], lb, ub)
                
                # Cập nhật fitness
                empire_colonies[i]['fit'] = np.apply_along_axis(self.problem.fitness, 1, empire_colonies[i]['pos'])
                
                # C. POS EXCHANGE (Đảo chính)
                # Nếu thuộc địa giỏi hơn Vua, hoán đổi vị trí
                best_col_idx = np.argmin(empire_colonies[i]['fit'])
                if empire_colonies[i]['fit'][best_col_idx] < empires_fit[i]:
                    # Swap
                    empires[i], empire_colonies[i]['pos'][best_col_idx] = empire_colonies[i]['pos'][best_col_idx].copy(), empires[i].copy()
                    empires_fit[i], empire_colonies[i]['fit'][best_col_idx] = empire_colonies[i]['fit'][best_col_idx], empires_fit[i]

            # Cập nhật Global Best
            min_emp_idx = np.argmin(empires_fit)
            if empires_fit[min_emp_idx] < self.global_best_fitness:
                self.update_global_best(empires[min_emp_idx], empires_fit[min_emp_idx])

            # D. IMPERIALISTIC COMPETITION (Cạnh tranh đế quốc)
            # Tính Total Cost (Vua + 0.1 * Trung bình dân)
            total_costs = np.zeros(self.n_empires)
            zeta = 0.1
            
            for i in range(self.n_empires):
                if len(empire_colonies[i]['fit']) > 0:
                    total_costs[i] = empires_fit[i] + zeta * np.mean(empire_colonies[i]['fit'])
                else:
                    total_costs[i] = empires_fit[i]
            
            # Tìm đế quốc yếu nhất
            weakest_emp_idx = np.argmax(total_costs)
            
            # Nếu đế quốc yếu còn thuộc địa để cướp
            if len(empire_colonies[weakest_emp_idx]['pos']) > 0:
                # Lấy thuộc địa yếu nhất của nước yếu nhất
                weakest_col_idx = np.argmax(empire_colonies[weakest_emp_idx]['fit'])
                col_pos = empire_colonies[weakest_emp_idx]['pos'][weakest_col_idx].copy()
                col_fit = empire_colonies[weakest_emp_idx]['fit'][weakest_col_idx]
                
                # Xóa khỏi nước yếu
                empire_colonies[weakest_emp_idx]['pos'] = np.delete(empire_colonies[weakest_emp_idx]['pos'], weakest_col_idx, axis=0)
                empire_colonies[weakest_emp_idx]['fit'] = np.delete(empire_colonies[weakest_emp_idx]['fit'], weakest_col_idx)
                
                # Chọn nước thắng (Roulette Wheel)
                comp_powers = np.max(total_costs) - total_costs
                if np.sum(comp_powers) == 0:
                    probs = np.ones(self.n_empires) / self.n_empires
                else:
                    probs = comp_powers / np.sum(comp_powers)
                
                winning_idx = np.random.choice(range(self.n_empires), p=probs)
                
                # Thêm vào nước thắng
                empire_colonies[winning_idx]['pos'] = np.vstack([empire_colonies[winning_idx]['pos'], col_pos])
                empire_colonies[winning_idx]['fit'] = np.append(empire_colonies[winning_idx]['fit'], col_fit)

            self.save_history()

        return self.global_best_solution, self.global_best_fitness