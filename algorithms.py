import numpy as np
import copy
class BaseAllocator:
    """所有分配算法的基类 (Interface)"""
    def allocate(self, drones, tasks):
        raise NotImplementedError("子类必须实现具体的分配逻辑")

class SSIAuction(BaseAllocator):
    """顺序单项拍卖算法 (SSI) 的具体实现"""
    def allocate(self, drones, tasks):
        print("--- 开始 SSI 拍卖过程 ---")
        # 筛选出未分配的任务
        unassigned_tasks = [t for t in tasks if not t.assigned]

        while unassigned_tasks:
            best_bid = float('inf')
            best_pair = None  # (drone_obj, task_obj)

            # 遍历寻找全局最优匹配
            for drone in drones:
                for task in unassigned_tasks:
                    dist = np.linalg.norm(drone.pos - task.pos)
                    if dist < best_bid:
                        best_bid = dist
                        best_pair = (drone, task)

            # 锁定胜出者并更新状态
            if best_pair:
                winner_drone, won_task = best_pair
                won_task.assigned = True
                winner_drone.assign_task(won_task, best_bid)
                unassigned_tasks.remove(won_task) # 移出待分配列表
                
                print(f"任务 {won_task.id:2d} -> 无人机 {winner_drone.color:5s} (成本: {best_bid:.2f})")
                
class CBBA_Allocator(BaseAllocator):
    """基于共识的捆绑拍卖算法 (CBBA)"""
    
    def __init__(self, max_tasks=4, lambda_discount=0.95):
        # 算法特有超参数通过初始化传入
        self.L_T = max_tasks
        self.LAMBDA = lambda_discount

    def _calc_path_score(self, drone_pos, path_task_ids, tasks):
        """内部方法：计算时间折扣总得分 S_i^{p_i}"""
        if not path_task_ids: return 0
        score = 0
        current_pos = drone_pos
        time_elapsed = 0
        
        for t_id in path_task_ids:
            task = tasks[t_id]
            dist = np.linalg.norm(current_pos - task.pos)
            time_elapsed += dist / 10.0 # 假设速度 10m/s
            score += (self.LAMBDA ** time_elapsed) * task.score
            current_pos = task.pos
        return score

    def allocate(self, drones, tasks):
        n_drones = len(drones)
        n_tasks = len(tasks)

        # [cite_start]核心状态矩阵初始化 [cite: 198]
        y = np.zeros((n_drones, n_tasks))
        z = np.full((n_drones, n_tasks), -1)
        bundles = [[] for _ in range(n_drones)]
        paths = [[] for _ in range(n_drones)]

        converged = False
        iteration = 0

        print("--- 开始 CBBA 分配迭代 ---")
        while not converged:
            iteration += 1
            bundles_old = copy.deepcopy(bundles)
            
            # [cite_start]--- Phase 1: Bundle Construction (第一阶段：构建任务包) [cite: 178, 179] ---
            for i, drone in enumerate(drones):
                while len(bundles[i]) < self.L_T:
                    best_task = -1
                    best_marginal_score = 0
                    best_insert_idx = -1
                    
                    for j, task in enumerate(tasks):
                        if j in bundles[i]: continue
                        
                        max_score_gain = 0
                        best_idx = -1
                        current_path_score = self._calc_path_score(drone.pos, paths[i], tasks)
                        
                        # 评估插入最佳位置
                        for insert_idx in range(len(paths[i]) + 1):
                            new_path = paths[i][:insert_idx] + [j] + paths[i][insert_idx:]
                            new_score = self._calc_path_score(drone.pos, new_path, tasks)
                            marginal_score = new_score - current_path_score
                            
                            if marginal_score > max_score_gain:
                                max_score_gain = marginal_score
                                best_idx = insert_idx
                        
                        if max_score_gain > y[i][j] and max_score_gain > best_marginal_score:
                            best_marginal_score = max_score_gain
                            best_task = j
                            best_insert_idx = best_idx
                    
                    if best_task == -1: break
                    
                    # [cite_start]添加任务 [cite: 197]
                    bundles[i].append(best_task)
                    paths[i].insert(best_insert_idx, best_task)
                    y[i][best_task] = best_marginal_score
                    z[i][best_task] = i

            # [cite_start]--- Phase 2: Conflict Resolution (第二阶段：冲突解决) [cite: 178, 203] ---
            y_global_max = np.max(y, axis=0)
            z_global_max = np.argmax(y, axis=0) # 上次修复的 Bug
            
            for i in range(n_drones):
                outbid = False
                outbid_idx = -1
                
                for j in range(n_tasks):
                    if y_global_max[j] > y[i][j]:
                        y[i][j] = y_global_max[j]
                        z[i][j] = z_global_max[j]
                        
                for idx, t_id in enumerate(bundles[i]):
                    if z[i][t_id] != i:
                        outbid = True
                        outbid_idx = idx
                        break
                        
                # [cite_start]释放被抢走任务及之后的所有任务 [cite: 211, 212]
                if outbid:
                    dropped_tasks = bundles[i][outbid_idx:]
                    for t_id in dropped_tasks:
                        if z[i][t_id] == i:
                            y[i][t_id] = 0
                            z[i][t_id] = -1
                    bundles[i] = bundles[i][:outbid_idx]
                    paths[i] = [t for t in paths[i] if t not in dropped_tasks]
                    
            if bundles == bundles_old:
                converged = True

        print(f">> 收敛完成！共消耗 {iteration} 轮迭代。\n")

        # --- 最终阶段：将 CBBA 算出的分配结果写入真正的 Drone 和 Task 对象中 ---
        for i, drone in enumerate(drones):
            # 此时的路径存储的是 Task 的索引，依次为无人机实际分配
            current_pos = drone.pos.copy()
            for t_id in paths[i]:
                task_obj = tasks[t_id]
                task_obj.assigned = True
                dist = np.linalg.norm(current_pos - task_obj.pos)
                drone.assign_task(task_obj, dist)
                current_pos = task_obj.pos.copy()