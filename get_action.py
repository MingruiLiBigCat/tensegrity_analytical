import numpy as np
from scipy.optimize import minimize, Bounds
import os
import csv
import matplotlib.pyplot as plt

class TensegrityStructure:
    def __init__(self, node_positions, rod_pairs,
                 elastic_cable_pairs, rigid_cable_pairs,
                 rest_lengths, stiffness, mass, fixed_nodes):
        self.node_positions = node_positions.copy()
        self.rod_pairs = rod_pairs
        self.elastic_cable_pairs = elastic_cable_pairs  # passive
        self.rigid_cable_pairs = rigid_cable_pairs      # active (controlled)
        self.rest_lengths = rest_lengths  # Only for rigid cables
        self.stiffness = stiffness        # Only for elastic cables
        self.mass = mass
        if fixed_nodes == [-1, -1, -1]:
            lowest_z_indices = np.argsort(node_positions[:, 2])[:2].tolist()
            self.fixed_nodes = lowest_z_indices
            #print(f"⚠️ 检测到默认 fixed_nodes，已自动设为最低 2 个点: {self.fixed_nodes}")
        else:
            self.fixed_nodes = fixed_nodes
        self.g = np.array([0, 0, -9.81])

        #print("🔧 Rods:", self.rod_pairs)
        #print("🔧 Fixed nodes:", self.fixed_nodes)
        #print("🔧 Rigid cables:", self.rigid_cable_pairs)

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
        for k, (i, j) in enumerate(self.elastic_cable_pairs):
            L = np.linalg.norm(nodes[i] - nodes[j])
            delta = L - self.stiffness[k][1]
            k_val = self.stiffness[k][0]
            P_e += 0.5 * k_val * delta ** 2
        return P_g + P_e

    def rod_constraints(self, x):
        nodes = self.unpack(x)
        constraints = []
        for i, j in self.rod_pairs:
            v = nodes[i] - nodes[j]
            L = np.linalg.norm(v)
            L0 = np.linalg.norm(self.node_positions[i] - self.node_positions[j])
            #print(f"🧵 Rod [{i}-{j}] | Current: {L:.4f}, Target: {L0:.4f}, Error: {L - L0:.4e}")
            constraints.append(L - L0)
        for k in self.fixed_nodes:
            diff = (nodes[k] - self.node_positions[k]).tolist()
            #print(f"📌 Fixed node {k} delta: {diff}")
            constraints.extend(diff)
        for k, (i, j) in enumerate(self.rigid_cable_pairs):
            L = np.linalg.norm(nodes[i] - nodes[j])
            #print(f"🔩 Rigid cable [{i}-{j}] | Current: {L:.4f}, Target: {self.rest_lengths[k]:.4f}")
            constraints.append(L - self.rest_lengths[k])
        return np.array(constraints)

    def ground_constraint(self, x):
        nodes = self.unpack(x)
        min_z = nodes[:, 2].min()
        #print(f"🟢 Min z in ground constraint: {min_z:.4f}")
        return nodes[:, 2]

    def center_of_mass(self, nodes):
        total_mass = np.sum(self.mass)
        com = np.zeros(3)
        for k, (i, j) in enumerate(self.rod_pairs):
            rod_center = 0.5 * (nodes[i] + nodes[j])
            com += self.mass[k] * rod_center
        return com / total_mass

    def update_position_from_env(self, env):
        _, _, env_nodes = env._get_obs()
        self.node_positions = np.array(env_nodes).reshape(-1, 3)

# --- Forward kinematics with diagnostics ---
def forward_kinematics_trust_verbose_fixed(structure):
    x0 = structure.pack(structure.node_positions)
    eq_constraint = {'type': 'eq', 'fun': structure.rod_constraints}
    ineq_constraint = {'type': 'ineq', 'fun': structure.ground_constraint}
    bounds = Bounds([-np.inf] * len(x0), [np.inf] * len(x0))

    print("📏 Num variables:", len(x0))
    print("🗐 Num equality constraints:", len(structure.rod_constraints(x0)))
    print("🗐 Num inequality constraints:", len(structure.ground_constraint(x0)))

    res = minimize(
        fun=structure.potential_energy,
        x0=x0,
        method='SLSQP',
        constraints=[eq_constraint, ineq_constraint],
        bounds=bounds,
        options={
            'maxiter': 10000,
            'ftol': 1e-9,
            'disp': True
        }
    )

    if not res.success:
        raise RuntimeError(f"Forward kinematics failed: {res.message}")

    return structure.unpack(res.x)

# --- 可视化结构保存图像 ---
def save_structure_plot(nodes, step, save_dir="figs"):
    os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color='blue')
    for i, pos in enumerate(nodes):
        ax.text(pos[0], pos[1], pos[2], str(i), color='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Step {step}")
    plt.savefig(os.path.join(save_dir, f"step_{step:03d}.png"))
    plt.close()

# --- 保存绳长历史为 CSV ---
def save_rest_lengths_csv(history, filename="rest_lengths.csv"):
    if not history:
        print("⚠️ rest_lengths_history 为空，跳过 CSV 保存")
        return
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"rigid_cable_{i}" for i in range(len(history[0]))])
        for row in history:
            writer.writerow(row)

# --- 单步 IK 更新函数（仅作用于刚性绳） ---
def ik_step(structure, q_current, com_target, history=None, step=None):
    structure.rest_lengths = q_current
    nodes = structure.node_positions
    current_com = structure.center_of_mass(nodes)
    error = com_target - current_com

    if history is not None:
        history.append(q_current.copy())
    if step is not None:
        save_structure_plot(nodes, step)

    if np.linalg.norm(error) < 1e-4:
        return q_current, nodes

    n = len(q_current)
    J = np.zeros((3, n))
    for i in range(n):
        dq = np.zeros_like(q_current)
        dq[i] = 1e-4
        structure.rest_lengths = q_current + dq
        ne_plus = structure.center_of_mass(forward_kinematics_trust_verbose_fixed(structure))
        J[:, i] = (ne_plus - current_com) / 1e-4

    structure.rest_lengths = q_current
    dq = np.linalg.pinv(J) @ error
    q_next = q_current + dq
    structure.rest_lengths = q_next
    nodes = forward_kinematics_trust_verbose_fixed(structure)
    return q_next, nodes
