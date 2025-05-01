import tr_env_gym
import os
import numpy as np
from get_action import *
from planning import COM_scheduler
import cv2
import signal
import sys
import csv

# Global variables for interrupt handling
rest_lengths_history = []
interrupted = False
out = None

# Handle Ctrl+C (SIGINT): save data and release video writer
def signal_handler(sig, frame):
    global interrupted, out
    interrupted = True
    print("\nInterrupt signal received. Preparing to save data...")
    if out is not None:
        out.release()
        print("Video writer closed")
    save_rest_lengths_csv(rest_lengths_history)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def save_rest_lengths_csv(history, filename="rest_lengths.csv"):
    if not history:
        print("rest_lengths_history is empty, skipping CSV export")
        return
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + [f"rigid_cable_{i}" for i in range(6)] + ["com_x", "com_y", "target_com_x", "target_com_y"])
        for row in history:
            writer.writerow(row)
    print("rest_lengths.csv saved successfully")

def test_frame(env):
    frame = env.render()
    cv2.imwrite(f'frames/start.png', frame)
    state, observation, global_obs = env._get_obs()
    print("observation:", global_obs)
    exit(1)

def run():
    # 1. Initialize environment and COM scheduler
    env = tr_env_gym.tr_env_gym(
        render_mode="human",
        width=640,
        height=480,
        xml_file=os.path.join(os.getcwd(), "t.xml"),
        is_test=False,
        desired_action="straight",
        desired_direction=1,
        terminate_when_unhealthy=True
    )
    scheduler = COM_scheduler(1, 0)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter('output.mp4', fourcc, 60, (640, 480))
    os.makedirs('frames', exist_ok=True)

    # 2. Get initial data from environment
    env.render()
    _, _, env_nodes = env._get_obs()
    env_nodes = np.array(env_nodes).reshape(-1, 3)
    rod_pairs = env.get_rod_pairs()
    elastic_cables = env.get_spring_pairs()
    rigid_cables = env.get_cable_pairs()
    rest_lengths = np.array(env.get_rest_lengths())
    stiffness = np.array(env.get_stiffnesses())
    mass = np.array(env.get_rod_masses())
    fixed_nodes = env.get_fixed_nodes()

    structure = TensegrityStructure(
        node_positions=np.array(env_nodes),
        rod_pairs=rod_pairs,
        rigid_cable_pairs=rigid_cables,
        elastic_cable_pairs=elastic_cables,
        rest_lengths=rest_lengths,
        stiffness=stiffness,
        mass=mass,
        fixed_nodes=fixed_nodes
    )
    structure.node_positions += 1e-4 * np.random.randn(*structure.node_positions.shape)

    try:
        print("Performing initial structural correction...")
        corrected_nodes = forward_kinematics_trust_verbose_fixed(structure)
        structure.node_positions = corrected_nodes.copy()
        print("Initial structure corrected successfully")
    except RuntimeError as e:
        print("Initial correction failed:", e)
        print("Fallback enabled: using original node positions")

    q_current = rest_lengths.copy()[:6]
    os.makedirs("figs", exist_ok=True)

    q_current = structure.rest_lengths.copy()[:6]
    done = False

    # 4. Main control loop
    for step in range(500):
        print(f"Step {step}")
        structure.update_position_from_env(env)

        com = structure.center_of_mass(np.array(structure.node_positions))
        x0, y0 = com[0], com[1]
        fnodes = structure.get_fixed_nodes()
        fnodes = sort_fnodes(fnodes)
        if fnodes[0] == -1:
            print("Tipping detected, no action generated. Current COM:", com[:2])
            obs, done, info = env.step(np.array([0, 0, 0, 0, 0, 0]))
            env.render()
            rest_lengths_history.append(np.concatenate([[step], q_current, com, com]))
            continue

        x1, y1 = structure.node_positions[fnodes[0]][:2]
        x2, y2 = structure.node_positions[fnodes[1]][:2]
        x3, y3 = structure.node_positions[fnodes[2]][:2]
        
        # Get COM target from scheduler
        (x_target, y_target), _ = scheduler.get_COM(x0, y0, x1, y1, x2, y2, x3, y3)
        target_com = np.array([x_target, y_target, com[2] + 0.01])
        print(f"{step} Target COM: {target_com}, Current COM: {com}")
        
        # Run inverse kinematics step
        q_next, nodes = ik_step(structure, q_current, target_com, step=step)
        if nodes is not None:
            action = q_next - q_current

        if not done:
            for _ in range(2):
                obs, done, info = env.step(action[:6])
        env.render()

        # Record history
        rest_lengths_history.append(np.concatenate([[step], q_current, com, target_com]))

        q_current = q_next.copy()
        print(f"[step {step}] COM = {structure.center_of_mass(nodes)}")

        fig_path = os.path.join("figs", f"step_{step:03d}.png")
        save_structure_plot(nodes, step, save_dir="figs")

        if done:
            print("Simulation terminated")
            break

    out.release()
    print("Video writer released")
    if not interrupted:
        save_rest_lengths_csv(rest_lengths_history)

def sort_fnodes(fnodes):
    odd = [x for x in fnodes if x % 2 != 0]
    even = [x for x in fnodes if x % 2 == 0]
    
    if len(odd) == 2 and len(even) == 1:
        return odd + even
    elif len(even) == 2 and len(odd) == 1:
        return even + odd
    else:
        return fnodes

if __name__ == "__main__":
    run()
