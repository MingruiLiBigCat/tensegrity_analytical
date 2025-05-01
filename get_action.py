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
        self.rest_lengths = rest_lengths
        self.stiffness = stiffness
        self.mass = mass
        self.fixed_nodes = self.get_fixed_nodes()
        self.g = np.array([0, 0, -9.81])

    def get_fixed_nodes(self):
        sorted_position = np.sort(self.node_positions[:, 2])
        if sorted_position[2] - sorted_position[0] > 0.08:
            return [-1, -1, -1]
        lowest_z_indices = np.argsort(self.node_positions[:, 2])[:3].tolist()
        print("Fixed Nodes:", lowest_z_indices)
        self.fixed_nodes = lowest_z_indices
        return lowest_z_indices

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
            constraints.append(L - L0)
        for k in self.fixed_nodes:
            diff = (nodes[k] - self.node_positions[k]).tolist()
            constraints.extend(diff)
        for k, (i, j) in enumerate(self.rigid_cable_pairs):
            L = np.linalg.norm(nodes[i] - nodes[j])
            constraints.append(L - self.rest_lengths[k])
        return np.array(constraints)

    def ground_constraint(self, x):
        nodes = self.unpack(x)
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
        print(env_nodes)
        self.node_positions = np.array(env_nodes).reshape(-1, 3)

# Forward kinematics solver using constrained energy minimization
def forward_kinematics_trust_verbose_fixed(structure):
    x0 = structure.pack(structure.node_positions)
    eq_constraint = {'type': 'eq', 'fun': structure.rod_constraints}
    ineq_constraint = {'type': 'ineq', 'fun': structure.ground_constraint}
    bounds = Bounds([-np.inf] * len(x0), [np.inf] * len(x0))

    res = minimize(
        fun=structure.potential_energy,
        x0=x0,
        method='trust-constr',
        constraints=[eq_constraint, ineq_constraint],
        bounds=bounds,
        options={
            'maxiter': 30000,
            'gtol': 3e-3,
            'disp': False,
            'xtol': 3e-3
        }
    )

    if not res.success:
        print(f"Forward kinematics failed: {res.message}")
        return None

    return structure.unpack(res.x)

# Save 3D structure plot of tensegrity nodes
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

# Save rest length history to CSV
def save_rest_lengths_csv(history, filename="rest_lengths.csv"):
    if not history:
        print("rest_lengths_history is empty, skipping CSV save")
        return
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"rigid_cable_{i}" for i in range(len(history[0]))])
        for row in history:
            writer.writerow(row)

# Single-step inverse kinematics using finite difference Jacobian
def ik_step(structure, q_current, com_target, step=None):
    structure.rest_lengths = q_current
    nodes = structure.node_positions
    current_com = structure.center_of_mass(nodes)
    error = com_target - current_com

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
        perturbed_nodes = forward_kinematics_trust_verbose_fixed(structure)
        if perturbed_nodes is None:
            return None, None
        ne_plus = structure.center_of_mass(perturbed_nodes)
        J[:, i] = (ne_plus - current_com) / 1e-4

    structure.rest_lengths = q_current
    dq = np.linalg.pinv(J) @ error
    q_next = q_current + dq
    structure.rest_lengths = q_next
    nodes = forward_kinematics_trust_verbose_fixed(structure)
    if nodes is None:
        return None, None
    return q_next, nodes
