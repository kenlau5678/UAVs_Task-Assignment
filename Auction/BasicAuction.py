import numpy as np
import matplotlib.pyplot as plt

class UAV:
    def __init__(self, uav_id, start_pos):
        self.id = uav_id
        self.current_pos = np.array(start_pos)
        self.path = [np.array(start_pos)]  # 记录路径用于画图
        self.assigned_tasks = []           # 记录中标的任务ID
        self.total_cost = 0.0              # 累计飞行距离

    def calculate_bid(self, task_pos):
        """
        投标函数：计算从当前位置到任务位置的欧几里得距离
        """
        dist = np.linalg.norm(self.current_pos - task_pos)
        return dist

    def award_task(self, task, cost):
        """
        中标处理：更新位置，记录任务
        """
        self.assigned_tasks.append(task['id'])
        self.total_cost += cost
        
        # 核心逻辑：模拟无人机飞往该任务点，更新当前位置
        # 这样下一次竞拍时，是基于新位置计算距离的
        self.current_pos = task['pos']
        self.path.append(task['pos'])

class ContractNetProtocol:
    def __init__(self, num_uavs=3, num_tasks=10, area_size=100):
        self.area_size = area_size
        
        # 1. 初始化无人机 (随机位置)
        self.uavs = []
        for i in range(num_uavs):
            pos = np.random.rand(2) * area_size
            self.uavs.append(UAV(i, pos))
            
        # 2. 初始化任务点 (随机位置)
        self.tasks = []
        for i in range(num_tasks):
            self.tasks.append({
                'id': i,
                'pos': np.random.rand(2) * area_size
            })

    def run_auction(self):
        print(f"{'='*10} 开始拍卖 {'='*10}")
        
        # 简单的序贯拍卖：按任务列表顺序逐个拍卖
        # 注意：这种方法的缺点是结果受任务发布顺序影响（这是Level 1的局限性）
        for task in self.tasks:
            print(f"\n正在拍卖任务 ID: {task['id']} (位置: {task['pos'].round(1)})")
            
            bids = []
            
            # --- 招标与投标阶段 ---
            for uav in self.uavs:
                bid_price = uav.calculate_bid(task['pos'])
                bids.append((uav, bid_price))
                print(f"  -> UAV {uav.id} 报价 (距离): {bid_price:.2f}")
            
            # --- 赢家判定阶段 (Winner Determination) ---
            # 选择报价最低（距离最近）的无人机
            winner_uav, min_cost = min(bids, key=lambda x: x[1])
            
            # --- 签约阶段 ---
            winner_uav.award_task(task, min_cost)
            print(f"  *** 恭喜 UAV {winner_uav.id} 中标！成本: {min_cost:.2f}")

    def plot_results(self):
        plt.figure(figsize=(10, 8))
        plt.title("Level 1: Basic Contract Net Protocol (CNP) Assignment")
        plt.xlim(0, self.area_size)
        plt.ylim(0, self.area_size)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # 画出所有任务点
        for task in self.tasks:
            plt.scatter(task['pos'][0], task['pos'][1], c='black', marker='x', s=100)
            plt.text(task['pos'][0]+1, task['pos'][1]+1, f"T{task['id']}", fontsize=9)

        # 画出无人机轨迹
        for i, uav in enumerate(self.uavs):
            path = np.array(uav.path)
            c = colors[i % len(colors)]
            
            # 画路径线
            plt.plot(path[:, 0], path[:, 1], c=c, linestyle='--', alpha=0.6, label=f'UAV {uav.id}')
            
            # 画起点
            plt.scatter(path[0][0], path[0][1], c=c, marker='o', s=150, edgecolors='black', label=f'Start {uav.id}')
            
            # 标注中标任务顺序
            for idx, task_id in enumerate(uav.assigned_tasks):
                # 找到该任务的位置
                t_pos = self.tasks[task_id]['pos']
                # 在图上标出执行顺序
                plt.text(t_pos[0], t_pos[1]-4, f"[{idx+1}]", color=c, fontweight='bold')

        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()

# 运行模拟
if __name__ == "__main__":
    sim = ContractNetProtocol()
    sim.run_auction()
    sim.plot_results()