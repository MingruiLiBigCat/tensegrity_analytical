import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
import os
import csv

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
        self.fixed_nodes = fixed_nodes
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
        for k, (i, j) in enumerate(self.elastic_cable_pairs):
            L = np.linalg.norm(nodes[i] - nodes[j])
            delta = L - self.stiffness[k][1]  # Expected length is stored in stiffness[k][1]
            k_val = self.stiffness[k][0]
            P_e += 0.5 * k_val * delta ** 2
        return P_g + P_e

    def rod_constraints(self, x):
        nodes = self.unpack(x)
        constraints = []
        for i, j in self.rod_pairs:
            L = np.linalg.norm(nodes[i] - nodes[j])
            constraints.append(L - np.linalg.norm(self.node_positions[i] - self.node_positions[j]))
        for k in self.fixed_nodes:
            constraints.extend((nodes[k] - self.node_positions[k]).tolist())
        for k, (i, j) in enumerate(self.rigid_cable_pairs):
            L = np.linalg.norm(nodes[i] - nodes[j])
            constraints.append(L - self.rest_lengths[k])
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

# --- Trust-constr 前向运动学 ---
def forward_kinematics_trust_verbose_fixed(structure):
    x0 = structure.pack(structure.node_positions)
    eq_constraint = {'type': 'eq', 'fun': structure.rod_constraints}
    ineq_constraint = {'type': 'ineq', 'fun': structure.ground_constraint}
    bounds = Bounds([-np.inf] * len(x0), [np.inf] * len(x0))
    
    res = minimize(
        fun=structure.potential_energy,
        x0=x0,
        constraints=[eq_constraint, ineq_constraint],
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 1000000,
            'gtol': 1e-5,
            'xtol': 1e-6,
            'verbose': 0,
            'disp': False
        }
    )

    if not res.success and "Constraint violation" not in res.message:
        raise RuntimeError(f"Trust-constr optimization failed: {res.message}")
    elif not res.success:
        print(f"⚠️ Warning: Optimization terminated with status: {res.message}")

    return structure.unpack(res.x)

# --- 从 MuJoCo 环境中更新结构位置 ---
def update_position_from_env(structure, env):
    _,env_nodes = env._get_obs()
    structure.node_positions = np.array(env_nodes).reshape(-1,3)

# --- COM 轨迹规划 ---
def get_target_COM_from_scheduler(scheduler, env):
    foot_positions = env.get_foot_positions()
    foot, result = scheduler.get_COM(*foot_positions)
    return foot

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
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"rigid_cable_{i}" for i in range(len(history[0]))])
        for row in history:
            writer.writerow(row)

# --- 单步 IK 更新函数（仅作用于刚性绳） ---
def ik_step(structure, q_current, com_target, history=None, step=None):
    structure.rest_lengths = q_current
    nodes = forward_kinematics_trust_verbose_fixed(structure)
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