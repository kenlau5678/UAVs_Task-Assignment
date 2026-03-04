import numpy as np
from utils import Visualizer
from models import Drone, Task
from algorithms import BaseAllocator, CBBA_Allocator


if __name__ == "__main__":
    
    # 每次跑算法前，重新初始化实体
    drones = [
        Drone(drone_id=0, pos=[0, 0, 0], color='red'),
        Drone(drone_id=1, pos=[25, 0, 0], color='green'),
        Drone(drone_id=2, pos=[50, 0, 0], color='blue'),
        Drone(drone_id=3, pos=[75, 0, 0], color='orange'),
        Drone(drone_id=4, pos=[100, 0, 0], color='purple')
    ]
    tasks = [Task(task_id=i, pos=np.random.rand(3) * 100) for i in range(40)]
    allocator = CBBA_Allocator(max_tasks=10) # 切换为 CBBA，每架飞机最多拿10个任务
    
    # 算法执行（所有的接口都是一致的！）
    allocator.allocate(drones, tasks)

    # 打印与绘图
    print("\n--- 最终里程统计 ---")
    for d in drones:
        print(f"{d.color}队总里程: {d.total_dist:.1f}m, 分配任务数: {len(d.assigned_tasks)}")
        
    Visualizer.plot_3d(drones, tasks)