import numpy as np
from scipy.optimize import minimize

class TensegrityStructure:
    def __init__(self, node_positions, rod_pairs, cable_pairs, rest_lengths, stiffness, mass, fixed_nodes):
        # 初始化所有节点的三维坐标（形状为 N × 3）
        self.node_positions = node_positions.copy()

        # 刚性杆的连接对，每一项是一个二元组 (i, j)，表示节点 i 和节点 j 之间有一根刚性杆
        self.rod_pairs = rod_pairs

        # 弹性绳的连接对，每一项是一个二元组 (i, j)，表示节点 i 和节点 j 之间有一根弹性绳
        self.cable_pairs = cable_pairs

        # 每根弹性绳的静止长度（不受力状态下的自然长度）
        self.rest_lengths = rest_lengths

        # 每根弹性绳的刚度（反映弹性绳对拉伸的抵抗程度）
        self.stiffness = stiffness

        # 每根刚性杆的质量（用于重力势能计算）
        self.mass = mass

        # 被固定的节点编号列表，这些节点的位置在优化过程中保持不变（如连接地面）
        self.fixed_nodes = fixed_nodes

        # 重力加速度向量，默认沿 z 轴向下（单位：m/s²）
        self.g = np.array([0, 0, -9.81])
