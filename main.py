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
        writer.writerow(["step"]+[f"rigid_cable_{i}" for i in range(6)]+["com_x", "com_y", "target_com_x", "target_com_y"])
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
    # 1. 初始化环境与调度器
    env = tr_env_gym.tr_env_gym(
        render_mode="human",
        width=640,   # 添加宽度
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
    done = False
    # 4. 控制主循环
    for step in range(500):
        print(f"🚀 Step  {step}")
        structure.update_position_from_env(env)

        com = structure.center_of_mass(np.array(structure.node_positions))
        x0, y0 = com[0], com[1]
        fnodes = structure.get_fixed_nodes()
        fnodes = sort_fnodes(fnodes)
        if fnodes[0] == -1:
            print("Tipping, no action generated.COM position (x0, y0):", com[:2])
            obs, done,  info =env.step(np.array([0, 0, 0, 0, 0, 0]))
            env.render()
            rest_lengths_history.append(np.concatenate([[step], q_current, com, com]))
            continue
        x1, y1 = structure.node_positions[fnodes[0]][:2]
        x2, y2 = structure.node_positions[fnodes[1]][:2]
        x3, y3 = structure.node_positions[fnodes[2]][:2]
        # 由调度器给出目标 COM
        (x_target, y_target), _ = scheduler.get_COM(x0, y0, x1, y1, x2, y2, x3, y3)
        target_com = np.array([x_target, y_target,com[2]+0.01])
        print(f"{step} target of COM is: {target_com}, while now COM is{com}")
        # 执行单步 IK 解算
        q_next, nodes = ik_step(structure, q_current, target_com,  step=step)
        if nodes is not None:  
            action = q_next - q_current
        if done is not True:
            for _ in range(2):
                obs, done,  info = env.step(action[:6])
        env.render()
        #print(step, q_current, q_next, target_com)
        rest_lengths_history.append(np.concatenate([[step], q_current, com, target_com])) 
        # cv2.imwrite(f'frames/frame_{step:04d}.png', frame) 
        # out.write(frame)
        #out.release()

        q_current = q_next.copy()

        print(f"[step {step}] COM = {structure.center_of_mass(nodes)}")

        fig_path = os.path.join("figs", f"step_{step:03d}.png")
        save_structure_plot(nodes, step, save_dir="figs")

        if done:
            print("❗️Simulation Terminated")
            break

    out.release()
    print("📼 Video writer released")
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
