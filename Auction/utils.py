import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def plot_3d(drones, tasks):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 画任务点
        task_positions = np.array([t.pos for t in tasks])
        ax.scatter(task_positions[:, 0], task_positions[:, 1], task_positions[:, 2], 
                   c='black', marker='x', s=50, label='Tasks')

        # 画无人机轨迹
        for d in drones:
            path = np.array(d.path)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], c=d.color, label=f"Drone {d.color.title()}", linewidth=2)
            ax.scatter(path[0, 0], path[0, 1], path[0, 2], c=d.color, marker='o', s=100, edgecolors='black')
            ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], c=d.color, marker='^', s=80)

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title(f'3D Drone Auction Simulation (Tasks: {len(tasks)})')
        ax.legend()
        plt.show()