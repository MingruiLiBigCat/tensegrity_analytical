import tr_env_gym
import os
import numpy as np
from get_action import *
from planning import COM_scheduler
import cv2
import signal
import sys
import csv

# 全局变量以支持中断处理
rest_lengths_history = []
interrupted = False
out = None

# 中断信号处理：保存绳长历史、释放视频写入器
def signal_handler(sig, frame):
    global interrupted, out
    interrupted = True
    print("\n🚨 捕获中断信号，准备保存数据...")
    if out is not None:
        out.release()
        print("📼 已关闭视频写入器")
    save_rest_lengths_csv(rest_lengths_history)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def save_rest_lengths_csv(history, filename="rest_lengths.csv"):
    if not history:
        print("⚠️ rest_lengths_history 为空，跳过 CSV 保存")
        return
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"rigid_cable_{i}" for i in range(len(history[0]))])
        for row in history:
            writer.writerow(row)
    print("✅ 已保存 rest_lengths.csv")

def test_frame(env):
    frame = env.render()
    cv2.imwrite(f'frames/start.png', frame)
    state, observation, global_obs = env._get_obs()
    print("observation:", global_obs)
    exit(1)

def run():
    global out
    # 1. 初始化环境与调度器
    env = tr_env_gym.tr_env_gym(
        render_mode="rgb_array",
        xml_file=os.path.join(os.getcwd(), "3prism_jonathan_steady_side.xml"),
        is_test=False,
        desired_action="straight",
        desired_direction=1,
        terminate_when_unhealthy=True
    )
    scheduler = COM_scheduler(1, 0)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter('output.mp4', fourcc, 60, (640, 480))
    os.makedirs('frames', exist_ok=True)

    # 2. 获取初始化数据
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
        print("⏳ 正在进行结构初始校正...")
        corrected_nodes = forward_kinematics_trust_verbose_fixed(structure)
        structure.node_positions = corrected_nodes.copy()
        print("✅ 初始结构已校正完成")
    except RuntimeError as e:
        print("❌ 初始结构校正失败：", e)
        print("⚠️ 启用 fallback：直接使用原始节点继续运行")

    q_current = rest_lengths.copy()[:6]
    os.makedirs("figs", exist_ok=True)

    q_current = structure.rest_lengths.copy()[:6]

    # 4. 控制主循环
    for step in range(100):
        print(f"🚀 第 {step} 步")
        structure.update_position_from_env(env)

        com = structure.center_of_mass(np.array(structure.node_positions))
        x0, y0 = com[0], com[1]
        x1, y1 = structure.node_positions[0][:2]
        x2, y2 = structure.node_positions[3][:2]
        x3, y3 = structure.node_positions[4][:2]

        (x_target, y_target), _ = scheduler.get_COM(x0, y0, x1, y1, x2, y2, x3, y3)
        target_com = np.array([x_target, y_target, com[2]])

        q_next, nodes = ik_step(structure, q_current, target_com, history=rest_lengths_history, step=step)

        action = q_next - q_current
        obs, done, info = env.step(action[:6])
        frame = env.render()
        cv2.imwrite(f'frames/frame_{step:04d}.png', frame)
        out.write(frame)

        q_current = q_next.copy()

        print(f"[step {step}] COM = {structure.center_of_mass(nodes)}")

        fig_path = os.path.join("figs", f"step_{step:03d}.png")
        save_structure_plot(nodes, step, save_dir="figs")

        if done:
            print("❗️仿真终止")
            break

    out.release()
    print("📼 视频写入器已释放")
    if not interrupted:
        save_rest_lengths_csv(rest_lengths_history)

if __name__ == "__main__":
    run()
