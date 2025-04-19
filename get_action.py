import numpy as np
from scipy.optimize import minimize, Bounds

class TensegrityStructure:
    def __init__(self, node_positions, rod_pairs, cable_pairs, rest_lengths, stiffness, mass, fixed_nodes):
        self.node_positions = node_positions.copy()
        self.rod_pairs = rod_pairs
        self.cable_pairs = cable_pairs
        self.rest_lengths = rest_lengths
        self.stiffness = stiffness
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
        method='trust-constr',
        bounds=bounds,
        options={
            'maxiter': 10000,
            'gtol': 1e-5,
            'xtol': 1e-6,
            'verbose': 0,
            'disp': True
        }
    )

    if not res.success and "Constraint violation" not in res.message:
        raise RuntimeError(f"Trust-constr optimization failed: {res.message}")
    elif not res.success:
        print(f"⚠️ Warning: Optimization terminated with status: {res.message}")

    return structure.unpack(res.x)

# --- Jacobian 有限差分 ---
def jacobian_fd_trust(structure, q, delta=1e-4):
    n = len(q)
    J = np.zeros((3, n))
    structure.rest_lengths = q
    ne = structure.center_of_mass(forward_kinematics_trust_verbose_fixed(structure))
    for i in range(n):
        dq = np.zeros_like(q)
        dq[i] = delta
        structure.rest_lengths = q + dq
        ne_plus = structure.center_of_mass(forward_kinematics_trust_verbose_fixed(structure))
        J[:, i] = (ne_plus - ne) / delta
    structure.rest_lengths = q
    return J

# --- 逆向运动学主函数 ---
def inverse_kinematics_trust_fully_consistent(structure, q0, ne_target, tol=1e-4, max_iter=10):
    q = q0.copy()
    for _ in range(max_iter):
        structure.rest_lengths = q
        nodes = forward_kinematics_trust_verbose_fixed(structure)
        ne = structure.center_of_mass(nodes)
        error = ne_target - ne
        if np.linalg.norm(error) < tol:
            break
        G = jacobian_fd_trust(structure, q)
        dq = np.linalg.pinv(G) @ error
        q += dq
    return q, forward_kinematics_trust_verbose_fixed(structure)