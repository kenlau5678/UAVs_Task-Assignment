import numpy as np
import matplotlib.pyplot as plt
import copy

# ================= 配置参数 =================
AREA_SIZE = 100
NUM_UAVS = 3
NUM_TASKS = 15
UAV_VELOCITY = 10.0   # 无人机速度 m/s
UAV_MAX_RANGE = 300.0 # 最大航程 (米)
UAV_CAPACITY = 4      # 最大任务携带量 (个)
SERVICE_TIME = 2.0    # 每个任务需要耗时 (秒)
# ===========================================

class UAV:
    def __init__(self, uav_id, start_pos):
        self.id = uav_id
        self.start_pos = np.array(start_pos)
        self.path = [self.start_pos]  # 路径点坐标列表
        self.schedule = []            # 任务ID列表
        
        # 约束参数
        self.velocity = UAV_VELOCITY
        self.max_range = UAV_MAX_RANGE
        self.max_capacity = UAV_CAPACITY

    def get_path_metrics(self, path_points):
        """
        计算路径的总距离
        """
        dist = 0.0
        for i in range(len(path_points) - 1):
            dist += np.linalg.norm(path_points[i] - path_points[i+1])
        return dist

    def check_feasibility(self, path_points, schedule_ids, all_tasks_dict):
        """
        [核心约束检查]
        检查一条建议的路径是否满足：载重、航程、时间窗
        """
        # 1. 检查载重 (Capacity)
        # path_points 包含起点，所以任务数是 len - 1
        num_tasks = len(path_points) - 1
        if num_tasks > self.max_capacity:
            return False, "Capacity Full"

        # 2. 检查航程 (Range)
        total_dist = self.get_path_metrics(path_points)
        if total_dist > self.max_range:
            return False, "Out of Fuel"

        # 3. 检查时间窗 (Time Windows)
        # 逻辑：到达时间 + 等待时间 -> 开始时间 -> 结束时间
        current_time = 0.0  # 假设 T=0 出发
        current_pos = path_points[0] # 起点

        # 遍历路径中的每一个任务 (跳过 path[0] 起点)
        for i, task_pos in enumerate(path_points[1:]):
            task_id = schedule_ids[i]
            task_info = all_tasks_dict[task_id]
            
            # 飞行时间
            dist = np.linalg.norm(task_pos - current_pos)
            flight_time = dist / self.velocity
            
            arrival_time = current_time + flight_time
            
            # [约束] 如果到达时间已经超过了最晚开始时间 -> 迟到了，路径无效
            if arrival_time > task_info['late_start']:
                return False, "Time Window Violation (Late)"
            
            # 如果早到了，需要等待 (Wait)
            start_time = max(arrival_time, task_info['early_start'])
            
            # 完成任务并准备去下一个
            finish_time = start_time + task_info['service_time']
            
            # 更新状态
            current_time = finish_time
            current_pos = task_pos

        return True, "OK"

    def calculate_marginal_cost(self, new_task, all_tasks_dict):
        """
        插入启发式 + 约束检查
        """
        best_cost = float('inf')
        best_insert_idx = -1
        
        current_dist = self.get_path_metrics(self.path)
        
        # 尝试插入到现有路径的每一个可能位置
        # range(1, len+1) 表示可以插在第一个任务前，也可以插在最后
        for i in range(1, len(self.path) + 1):
            # --- 构造临时方案 ---
            temp_path = copy.deepcopy(self.path)
            temp_path.insert(i, new_task['pos'])
            
            temp_schedule = copy.deepcopy(self.schedule)
            temp_schedule.insert(i - 1, new_task['id'])
            
            # --- 步骤 1: 约束检查 (Feasibility Check) ---
            is_valid, reason = self.check_feasibility(temp_path, temp_schedule, all_tasks_dict)
            
            if is_valid:
                # --- 步骤 2: 如果合法，计算边际成本 ---
                new_dist = self.get_path_metrics(temp_path)
                marginal_cost = new_dist - current_dist
                
                if marginal_cost < best_cost:
                    best_cost = marginal_cost
                    best_insert_idx = i
            else:
                # 路径不合法 (比如会导致后续任务超时)，直接跳过
                pass
                
        return best_cost, best_insert_idx

    def assign_task(self, task, insert_idx):
        self.path.insert(insert_idx, task['pos'])
        self.schedule.insert(insert_idx - 1, task['id'])

class SSIAuctionWithConstraints:
    def __init__(self):
        # 1. 初始化无人机
        self.uavs = [UAV(i, np.random.rand(2) * AREA_SIZE) for i in range(NUM_UAVS)]
        
        # 2. 初始化任务 (增加时间窗属性)
        self.tasks = []
        self.tasks_dict = {} #用于快速查找
        
        for i in range(NUM_TASKS):
            # 随机生成时间窗：[earliest, latest]
            # 确保时间窗合理，不要太早导致无法完成
            t_early = np.random.uniform(0, 30) 
            t_late = t_early + np.random.uniform(10, 30) # 窗口宽度至少10秒
            
            task = {
                'id': i,
                'pos': np.random.rand(2) * AREA_SIZE,
                'early_start': t_early,
                'late_start': t_late,
                'service_time': SERVICE_TIME
            }
            self.tasks.append(task)
            self.tasks_dict[i] = task
            
        self.unassigned_tasks = self.tasks.copy()
        self.abandoned_tasks = [] # 记录那些没人能做的任务

    def run_auction(self):
        print(f"{'='*10} 开始 SSI 拍卖 (带约束: 载重/航程/时间窗) {'='*10}")
        
        round_num = 1
        
        while self.unassigned_tasks:
            best_global_bid = float('inf')
            winning_uav = None
            winning_task = None
            winning_insert_idx = -1
            
            # --- 全局竞标 ---
            for task in self.unassigned_tasks:
                for uav in self.uavs:
                    # 传入 tasks_dict 以便内部查询时间窗
                    cost, idx = uav.calculate_marginal_cost(task, self.tasks_dict)
                    
                    if cost < best_global_bid:
                        best_global_bid = cost
                        winning_uav = uav
                        winning_task = task
                        winning_insert_idx = idx
            
            # --- 分配结果 ---
            if winning_uav and winning_task:
                print(f"R{round_num}: UAV-{winning_uav.id} -> T{winning_task['id']} "
                      f"(位置 {winning_insert_idx}), 成本 {best_global_bid:.1f}")
                winning_uav.assign_task(winning_task, winning_insert_idx)
                self.unassigned_tasks.remove(winning_task)
            else:
                # 关键修复：如果所有无人机对所有剩余任务的报价都是 Inf
                # 说明剩下的任务根本没法做（违反约束），必须强制跳出
                print(f"R{round_num}: 剩余 {len(self.unassigned_tasks)} 个任务无法分配 (约束限制).")
                self.abandoned_tasks = self.unassigned_tasks
                break
                
            round_num += 1

    def plot_results(self):
        plt.figure(figsize=(10, 8))
        plt.title(f"Level 3: SSI with Constraints\n(Range={UAV_MAX_RANGE}, Cap={UAV_CAPACITY}, TimeWindow=Yes)")
        plt.xlim(0, AREA_SIZE)
        plt.ylim(0, AREA_SIZE)
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # 1. 画已分配的任务和路径
        for i, uav in enumerate(self.uavs):
            path = np.array(uav.path)
            c = colors[i % len(colors)]
            
            # 路径线
            if len(path) > 1:
                plt.plot(path[:, 0], path[:, 1], c=c, linestyle='-', alpha=0.6, label=f'UAV {uav.id}')
            
            # 起点
            plt.scatter(path[0][0], path[0][1], c=c, marker='s', s=120, edgecolors='black')

            # 任务点顺序和ID
            for seq, t_id in enumerate(uav.schedule):
                t_pos = self.tasks_dict[t_id]['pos']
                # 画任务点
                plt.scatter(t_pos[0], t_pos[1], c='white', edgecolors=c, marker='o', s=100)
                plt.text(t_pos[0], t_pos[1], f"T{t_id}", fontsize=8, ha='center', va='center', color='black')
                # 画序号
                plt.text(t_pos[0]+1, t_pos[1]+2, f"[{seq+1}]", fontsize=8, color=c, fontweight='bold')

        # 2. 画无法分配的任务 (灰色 X)
        for task in self.abandoned_tasks:
            plt.scatter(task['pos'][0], task['pos'][1], c='gray', marker='x', s=60)
            plt.text(task['pos'][0], task['pos'][1], f"T{task['id']}(Fail)", fontsize=8, color='gray')

        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.show()

if __name__ == "__main__":
    sim = SSIAuctionWithConstraints()
    sim.run_auction()
    sim.plot_results()