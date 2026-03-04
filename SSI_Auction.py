import numpy as np
from utils import Visualizer
from models import Drone, Task
from algorithms import BaseAllocator, SSIAuction


if __name__ == "__main__":
    
    # 1. 初始化实体对象
    drones = [
        Drone(drone_id=0, pos=[0, 0, 0], color='red'),
        Drone(drone_id=1, pos=[50, 0, 0], color='green'),
        Drone(drone_id=2, pos=[100, 0, 0], color='blue')
    ]
    
    tasks = [Task(task_id=i, pos=np.random.rand(3) * 100) for i in range(30)]

    # 2. 实例化分配器并执行 (⭐ 这里就是策略模式的精髓所在)
    allocator = SSIAuction()
    # 如果将来有新算法，只需改为： allocator = CBBA_Auction()
    allocator.allocate(drones, tasks)

    # 3. 打印结果与可视化
    print("\n--- 最终里程统计 ---")
    for d in drones:
        print(f"{d.color}队总里程: {d.total_dist:.1f}m, 分配任务数: {len(d.assigned_tasks)}")
        
    Visualizer.plot_3d(drones, tasks)