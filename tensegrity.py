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

    def pack(self, nodes):
        return nodes.flatten()

    def unpack(self, x):
        return x.reshape(-1, 3)

    def potential_energy(self, x):
        nodes = self.unpack(x)
        P_g = 0
        for k, (i, j) in enumerate(self.rod_pairs):
            r_cm = 0.5 * (nodes[i] + nodes[j])
            P_g += -self.mass[k] * self.g @ r_cm
        P_e = 0
        for k, (i, j) in enumerate(self.cable_pairs):
            L = np.linalg.norm(nodes[i] - nodes[j])
            delta = L - self.rest_lengths[k]
            P_e += 0.5 * self.stiffness[k] * delta ** 2
        return P_g + P_e

    def rod_constraints(self, x):
        nodes = self.unpack(x)
        constraints = []
        for i, j in self.rod_pairs:
            L = np.linalg.norm(nodes[i] - nodes[j])
            constraints.append(L - np.linalg.norm(self.node_positions[i] - self.node_positions[j]))
        for k in self.fixed_nodes:
            constraints.extend((nodes[k] - self.node_positions[k]).tolist())
        return np.array(constraints)

    def ground_constraint(self, x):
        nodes = self.unpack(x)
        return nodes[:, 2]  # z >= 0

    def center_of_mass(self, nodes):
        total_mass = np.sum(self.mass)
        com = np.zeros(3)
        for k, (i, j) in enumerate(self.rod_pairs):
            rod_center = 0.5 * (nodes[i] + nodes[j])
            com += self.mass[k] * rod_center
        return com / total_mass

def forward_kinematics(structure):
    x0 = structure.pack(structure.node_positions)
    constraints = [
        {'type': 'eq', 'fun': structure.rod_constraints},
        {'type': 'ineq', 'fun': structure.ground_constraint}
    ]
    res = minimize(
        fun=structure.potential_energy,
        x0=x0,
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 500, 'ftol': 1e-6}
    )
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    return structure.unpack(res.x)

def jacobian_fd(structure, q, delta=1e-4):
    M = len(q)
    G = []
    for i in range(M):
        dq = np.zeros_like(q)
        dq[i] = delta
        structure.rest_lengths = q + dq
        ne_plus = structure.center_of_mass(forward_kinematics(structure))
        structure.rest_lengths = q
        ne = structure.center_of_mass(forward_kinematics(structure))
        G.append((ne_plus - ne) / delta)
    return np.column_stack(G)

def inverse_kinematics(structure, q0, ne_target, tol=1e-4, max_iter=20):
    q = q0.copy()
    for _ in range(max_iter):
        structure.rest_lengths = q
        nodes = forward_kinematics(structure)
        ne = structure.center_of_mass(nodes)
        error = ne_target - ne
        if np.linalg.norm(error) < tol:
            break
        G = jacobian_fd(structure, q)
        dq = np.linalg.pinv(G) @ error
        q += dq
    return q, forward_kinematics(structure)
