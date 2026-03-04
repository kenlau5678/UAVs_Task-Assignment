#序贯单项拍卖 (Sequential Single-Item Auction, SSI)
import numpy as np
import matplotlib.pyplot as plt
import copy

class UAV:
    def __init__(self, uav_id, start_pos):
        self.id = uav_id
        self.start_pos = np.array(start_pos)
        # 路径列表：初始只有起点
        self.path = [self.start_pos] 
        # 任务ID列表：初始为空 (对应 path[1:] 的任务)
        self.schedule = [] 

    def get_total_distance(self, path_points):
        """计算给定路径点的总欧几里得距离"""
        dist = 0.0
        for i in range(len(path_points) - 1):
            dist += np.linalg.norm(path_points[i] - path_points[i+1])
        return dist

    def calculate_marginal_cost(self, task):
        """
        核心算法：插入启发式 (Insertion Heuristic)
        尝试将任务插入到当前路径的每一个可能位置，
        找出增加距离最少的位置。
        """
        best_cost = float('inf')
        best_insert_idx = -1
        
        current_total_dist = self.get_total_distance(self.path)
        
        # 尝试插入位置：从 1 到 len(path)
        # 注意：索引0是起点，不能插在起点前面
        for i in range(1, len(self.path) + 1):
            # 创建临时路径进行测试
            temp_path = copy.deepcopy(self.path)
            temp_path.insert(i, task['pos'])
            
            # 计算新路径总距离
            new_total_dist = self.get_total_distance(temp_path)
            
            # 边际成本 = 新总距离 - 旧总距离
            marginal_cost = new_total_dist - current_total_dist
            
            if marginal_cost < best_cost:
                best_cost = marginal_cost
                best_insert_idx = i
                
        return best_cost, best_insert_idx

    def assign_task(self, task, insert_idx):
        """正式将任务插入到指定位置"""
        self.path.insert(insert_idx, task['pos'])
        # schedule 对应的索引需要减1，因为 path 包含起点
        self.schedule.insert(insert_idx - 1, task['id'])

class SSIAuction:
    def __init__(self, num_uavs=3, num_tasks=10, area_size=100):
        self.area_size = area_size
        self.uavs = [UAV(i, np.random.rand(2) * area_size) for i in range(num_uavs)]
        self.tasks = [{'id': i, 'pos': np.random.rand(2) * area_size} for i in range(num_tasks)]
        self.unassigned_tasks = self.tasks.copy()

    def run_auction(self):
        print(f"{'='*10} 开始 SSI 拍卖 (插入启发式) {'='*10}")
        
        round_num = 1
        
        # 循环直到所有任务都被分配
        while self.unassigned_tasks:
            best_bid = float('inf')
            winning_uav = None
            winning_task = None
            winning_insert_idx = -1
            
            # --- 全局竞标阶段 ---
            # 每一轮，所有无人机对所有剩余任务进行报价
            # 找出“全局”成本最低的那个组合（贪婪策略的最优解）
            for task in self.unassigned_tasks:
                for uav in self.uavs:
                    cost, idx = uav.calculate_marginal_cost(task)
                    
                    # 如果这个组合比当前记录的还要好，记录下来
                    if cost < best_bid:
                        best_bid = cost
                        winning_uav = uav
                        winning_task = task
                        winning_insert_idx = idx
            
            # --- 中标与分配阶段 ---
            if winning_uav and winning_task:
                print(f"Round {round_num}: UAV {winning_uav.id} 赢得任务 T{winning_task['id']}. "
                      f"插入位置: {winning_insert_idx}. 边际成本: {best_bid:.2f}")
                
                winning_uav.assign_task(winning_task, winning_insert_idx)
                self.unassigned_tasks.remove(winning_task)
                round_num += 1

    def plot_results(self):
        plt.figure(figsize=(10, 8))
        plt.title("Level 2: SSI with Insertion Heuristic (Path Planning)")
        plt.xlim(0, self.area_size)
        plt.ylim(0, self.area_size)
        
        colors = ['red', 'blue', 'green']
        
        # 画任务点
        for task in self.tasks:
            plt.scatter(task['pos'][0], task['pos'][1], c='black', marker='x', s=100)
            plt.text(task['pos'][0]+1, task['pos'][1]+1, f"T{task['id']}", fontsize=10)

        # 画无人机路径
        for i, uav in enumerate(self.uavs):
            path = np.array(uav.path)
            c = colors[i % len(colors)]
            
            # 画线
            plt.plot(path[:, 0], path[:, 1], c=c, linestyle='-', marker='o', alpha=0.6, label=f'UAV {uav.id}')
            # 标记起点
            plt.scatter(path[0][0], path[0][1], c=c, marker='s', s=150, edgecolors='black', label=f'Start {uav.id}')
            
            # 在路径连线上标注访问顺序箭头
            # (这里简单起见，直接在任务点旁打印顺序)
            for seq_idx, task_id in enumerate(uav.schedule):
                # 找到该任务对象
                t_obj = next(t for t in self.tasks if t['id'] == task_id)
                plt.text(t_obj['pos'][0], t_obj['pos'][1]-5, f"[{seq_idx+1}]", color=c, fontweight='bold')

        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()

if __name__ == "__main__":
    sim = SSIAuction()
    sim.run_auction()
    sim.plot_results()