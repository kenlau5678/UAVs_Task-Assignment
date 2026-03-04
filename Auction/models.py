import numpy as np
class Drone:
    def __init__(self, drone_id, pos, color):
        self.id = drone_id
        self.pos = np.array(pos, dtype=float)
        self.color = color
        self.path = [self.pos.copy()] # 记录起始点
        self.total_dist = 0.0         # 记录总航程
        self.assigned_tasks = []      # 记录分配到的任务 ID

    def assign_task(self, task, cost):
        """无人机执行任务并更新自身状态"""
        self.total_dist += cost
        self.pos = task.pos.copy()
        self.path.append(self.pos.copy())
        self.assigned_tasks.append(task.id)

class Task:
    def __init__(self, task_id, pos,score=100):
        self.id = task_id
        self.pos = np.array(pos, dtype=float)
        self.assigned = False
        self.score = score