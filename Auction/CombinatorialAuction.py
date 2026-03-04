import numpy as np
import matplotlib.pyplot as plt
import itertools

# --- 1. 初始化设置 ---

drones = [
    {'id': 0, 'color': 'red',   'pos': np.array([0, 0, 0])},
    {'id': 1, 'color': 'green', 'pos': np.array([50, 0, 0])},
    {'id': 2, 'color': 'blue',  'pos': np.array([100, 0, 0])}
]

# 生成 6 个随机任务点 (为了计算速度，N 不宜过大)
num_tasks = 7
tasks = [{'id': i, 'pos': np.random.rand(3) * 100} for i in range(num_tasks)]

# --- 2. 核心函数：计算“任务包”的最短路径成本 (TSP) ---
def calculate_bundle_cost(start_pos, bundle_tasks):
    """
    计算从起点出发，遍历 bundle_tasks 中所有任务的最短路径（全排列暴力求解）
    """
    if not bundle_tasks:
        return 0, []
    
    min_cost = float('inf')
    best_order = []
    
    # 遍历任务执行的所有可能顺序 (Permutations)
    for perm in itertools.permutations(bundle_tasks):
        cost = 0
        current_pos = start_pos
        for task in perm:
            cost += np.linalg.norm(current_pos - task['pos'])
            current_pos = task['pos']
            
        if cost < min_cost:
            min_cost = cost
            best_order = perm
            
    return min_cost, list(best_order)

# --- 3. 组合拍卖：胜者决定问题 (WDP) ---
print(f"--- 开始组合拍卖 (任务数: {num_tasks}) ---")
print("正在计算所有可能的组合分配方案...")

best_global_cost = float('inf')
best_global_allocation = None
best_global_routes = None

# 生成所有可能的分配方案。每个任务可以分配给 3 架无人机中的任意一架
# 共有 3^6 = 729 种分配组合
all_possible_allocations = itertools.product(range(len(drones)), repeat=num_tasks)

for allocation in all_possible_allocations:
    current_global_cost = 0
    current_routes = {d['id']: [] for d in drones}
    
    # 1. 根据当前方案，将任务打包分给对应的无人机
    for drone_id in range(len(drones)):
        # 找出分配给当前无人机的任务包
        bundle = [tasks[i] for i in range(num_tasks) if allocation[i] == drone_id]
        
        # 2. 计算无人机执行该任务包的最短路径成本
        cost, route = calculate_bundle_cost(drones[drone_id]['pos'], bundle)
        
        current_global_cost += cost
        current_routes[drone_id] = route
        
    # 3. 记录全局成本最低的方案
    if current_global_cost < best_global_cost:
        best_global_cost = current_global_cost
        best_global_allocation = allocation
        best_global_routes = current_routes

print(f">> 拍卖完成！全局最低总成本: {best_global_cost:.2f}\n")

# --- 4. 准备绘图数据 ---
for d in drones:
    d['path'] = [d['pos']] # 加入起点
    route = best_global_routes[d['id']]
    for task in route:
        d['path'].append(task['pos'])
        print(f"无人机 {d['color']} 获得任务 {task['id']}")
    if not route:
        print(f"无人机 {d['color']} 未分配到任务")

# --- 5. 三维绘图 ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 画任务点
task_positions = np.array([t['pos'] for t in tasks])
ax.scatter(task_positions[:, 0], task_positions[:, 1], task_positions[:, 2], 
           c='black', marker='x', s=50, label='Tasks')
for t in tasks:
    ax.text(t['pos'][0], t['pos'][1], t['pos'][2]+3, f"T{t['id']}", fontsize=10)

# 画无人机轨迹
for d in drones:
    path = np.array(d['path'])
    if len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 
                c=d['color'], label=f"Drone {d['color'].title()}", linewidth=2, marker='o')
    # 标记起点
    ax.scatter(path[0, 0], path[0, 1], path[0, 2], c=d['color'], marker='s', s=100, edgecolors='black')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Combinatorial Auction for UAV Task Allocation')
ax.legend()
plt.show()