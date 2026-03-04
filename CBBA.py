import numpy as np
import matplotlib.pyplot as plt
import copy

# --- 1. 参数与环境初始化 ---
N_DRONES = 3
N_TASKS = 20
L_T = 7  # 每架无人机最多执行的任务数 (Lt) [cite: 88]
LAMBDA = 0.95  # 时间折扣因子 (\lambda) [cite: 291, 541]

drones = [
    {'id': 0, 'color': 'red',   'pos': np.array([0, 0, 0]),       'path': [], 'total_dist': 0},
    {'id': 1, 'color': 'green', 'pos': np.array([50, 0, 0]),     'path': [], 'total_dist': 0},
    {'id': 2, 'color': 'blue',  'pos': np.array([100, 0, 0]),    'path': [], 'total_dist': 0}
]
tasks = [{'id': j, 'pos': np.random.rand(3)*100, 'score': 100} for j in range(N_TASKS)]

# 核心状态向量初始化 [cite: 198]
y = np.zeros((N_DRONES, N_TASKS))      # winning bid list (出价列表)
z = np.full((N_DRONES, N_TASKS), -1)   # winning agent list (胜出者列表)
bundles = [[] for _ in range(N_DRONES)] # bi: 任务包
paths = [[] for _ in range(N_DRONES)]   # pi: 飞行路径

# --- 2. 核心计分函数：时间折扣奖励  ---
def calc_path_score(drone_pos, path_task_ids):
    """计算路径总得分 S_i^{p_i} [cite: 291]"""
    if not path_task_ids: return 0
    score = 0
    current_pos = drone_pos
    time_elapsed = 0
    
    for t_id in path_task_ids:
        dist = np.linalg.norm(current_pos - tasks[t_id]['pos'])
        time_elapsed += dist / 10.0 # 假设速度为 10m/s
        # 论文公式 (11): S = \sum \lambda^{\tau} * c [cite: 291]
        score += (LAMBDA ** time_elapsed) * tasks[t_id]['score']
        current_pos = tasks[t_id]['pos']
    return score

# --- 3. CBBA 主循环 ---
converged = False
iteration = 0

print("--- 开始 CBBA 分配迭代 ---")
while not converged:
    iteration += 1
    bundles_old = copy.deepcopy(bundles)
    
    # ---------------------------------------------------------
    # Phase 1: Bundle Construction (Algorithm 3) [cite: 179, 210]
    # ---------------------------------------------------------
    for i in range(N_DRONES):
        while len(bundles[i]) < L_T:
            best_task = -1
            best_marginal_score = 0
            best_insert_idx = -1
            
            # 遍历所有未在自己包里的任务
            for j in range(N_TASKS):
                if j in bundles[i]: continue
                
                # 计算将任务 j 插入路径的最佳位置和最大边际收益 [cite: 196, 210]
                max_score_gain = 0
                best_idx = -1
                for insert_idx in range(len(paths[i]) + 1):
                    new_path = paths[i][:insert_idx] + [j] + paths[i][insert_idx:]
                    new_score = calc_path_score(drones[i]['pos'], new_path)
                    marginal_score = new_score - calc_path_score(drones[i]['pos'], paths[i])
                    
                    if marginal_score > max_score_gain:
                        max_score_gain = marginal_score
                        best_idx = insert_idx
                
                # 只有出价高于当前全网最高价，才被认为是有效出价 (h_ij) [cite: 197, 210]
                if max_score_gain > y[i][j] and max_score_gain > best_marginal_score:
                    best_marginal_score = max_score_gain
                    best_task = j
                    best_insert_idx = best_idx
            
            # 如果没有可以带来正收益的有效任务，结束打包 [cite: 197]
            if best_task == -1: break
            
            # 更新状态 [cite: 197, 210]
            bundles[i].append(best_task)
            paths[i].insert(best_insert_idx, best_task)
            y[i][best_task] = best_marginal_score
            z[i][best_task] = i

# ---------------------------------------------------------
    # Phase 2: Conflict Resolution 
    # ---------------------------------------------------------
    # 模拟同步全连接网络通信 ：提前计算好全网针对每个任务的最高出价和真正的胜出者
    y_global_max = np.max(y, axis=0)
    z_global_max = np.argmax(y, axis=0) # BUG 修复：在更新本地 y 矩阵之前锁定胜者！
    
    for i in range(N_DRONES):
        outbid = False
        outbid_idx = -1
        
        # 1. 接收全网信息，更新本地认知 [cite: 169]
        for j in range(N_TASKS):
            if y_global_max[j] > y[i][j]:
                y[i][j] = y_global_max[j]
                z[i][j] = z_global_max[j]  # 直接应用真正的胜出者 ID
                
        # 2. 检查自己包里的任务是否被抢走 [cite: 211, 234]
        for idx, t_id in enumerate(bundles[i]):
            if z[i][t_id] != i: # 发现任务不属于自己了 [cite: 237]
                outbid = True
                outbid_idx = idx
                break
                
        # 3. 如果在 idx 处被截胡，释放该任务及之后的所有任务 [cite: 212, 234]
        if outbid:
            dropped_tasks = bundles[i][outbid_idx:]
            for t_id in dropped_tasks:
                if z[i][t_id] == i: # 只有之前认为是自己的才重置
                    y[i][t_id] = 0
                    z[i][t_id] = -1
            # 截断 bundle 和 path
            bundles[i] = bundles[i][:outbid_idx]
            paths[i] = [t for t in paths[i] if t not in dropped_tasks]
            
    # 检查收敛：如果所有的 bundle 都不再变化，则收敛 [cite: 335]
    if bundles == bundles_old:
        converged = True

print(f">> 收敛完成！共消耗 {iteration} 轮迭代。\n")

# --- 4. 打印与 3D 绘图 ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制任务
t_pos = np.array([t['pos'] for t in tasks])
ax.scatter(t_pos[:,0], t_pos[:,1], t_pos[:,2], c='black', marker='x', s=100, label='Tasks')
for t in tasks: ax.text(t['pos'][0], t['pos'][1], t_pos[t['id'],2]+2, f"T{t['id']}")

# 绘制无人机及路径
colors = ['red', 'green', 'blue']
for i in range(N_DRONES):
    print(f"无人机 {i} (颜色 {colors[i]}) 最终分配任务: {paths[i]}")
    route = [drones[i]['pos']] + [tasks[t]['pos'] for t in paths[i]]
    route = np.array(route)
    
    ax.scatter(route[0,0], route[0,1], route[0,2], c=colors[i], marker='s', s=100, edgecolors='k')
    if len(route) > 1:
        ax.plot(route[:,0], route[:,1], route[:,2], c=colors[i], linewidth=2, label=f'Drone {i}')

ax.set_title(f'CBBA 3D Trajectories (Iterations: {iteration})')
ax.legend()
plt.show()