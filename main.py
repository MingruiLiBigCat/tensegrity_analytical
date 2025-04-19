import tr_env_gym
import os
import numpy as np
from get_action import *
from planning import COM_scheduler


def run():
    # 1. 初始化环境与调度器
    env = tr_env_gym.tr_env_gym(
        render_mode="None",
        xml_file=os.path.join(os.getcwd(), "3prism_jonathan_steady_side.xml"),
        is_test=False,
        desired_action="straight",
        desired_direction=1,
        terminate_when_unhealthy=True
    )

    scheduler = COM_scheduler(1, 0)

    # 2. 获取初始化数据（只能通过接口）
    obs = env.reset()
    env_nodes = env.get_node_positions()
    rod_pairs = env.get_rod_pairs()
    cable_pairs = env.get_cable_pairs()
    rest_lengths = np.array(env.get_rest_lengths())
    stiffness = np.array(env.get_stiffnesses())
    mass = np.array(env.get_rod_masses())
    fixed_nodes = env.get_fixed_nodes()

    # 3. 初始化结构对象
    structure = TensegrityStructure(
        node_positions=np.array(env_nodes),
        rod_pairs=rod_pairs,
        cable_pairs=cable_pairs,
        rest_lengths=rest_lengths,
        stiffness=stiffness,
        mass=mass,
        fixed_nodes=fixed_nodes
    )

    q_current = rest_lengths.copy()
    rest_lengths_history = []

    os.makedirs("figs", exist_ok=True)

    # 4. 控制循环
    for step in range(100):
        update_position_from_env(structure, env)

        # 获取当前 COM 位置
        com = structure.center_of_mass(np.array(structure.node_positions))
        x0, y0 = com[0], com[1]

        # 获取三角底部三点的 x,y（通常是 node 0,1,2 的投影）
        x1, y1 = structure.node_positions[0][:2]
        x2, y2 = structure.node_positions[1][:2]
        x3, y3 = structure.node_positions[2][:2]

        # 目标 COM 来自 scheduler
        (x_target, y_target), debug_info = scheduler.get_COM(x0, y0, x1, y1, x2, y2, x3, y3)
        target_com = np.array([x_target, y_target, com[2]])  # 保持 z 不变

        # 单步 IK
        q_next, nodes = ik_step(structure, q_current, target_com, history=rest_lengths_history, step=step)

        action = q_next - q_current
        obs, reward, done, truncated, info = env.step(action)
        q_current = q_next.copy()

        print(f"[step {step}] COM = {structure.center_of_mass(nodes)}")

        # 可视化保存
        fig_path = os.path.join("figs", f"step_{step:03d}.png")
        plot_structure(nodes, structure.cable_pairs, structure.rod_pairs, save_path=fig_path)

        if done or truncated:
            print("❗️仿真终止")
            break

    # 5. 导出 CSV
    save_rest_lengths_csv(rest_lengths_history, filename="rest_lengths.csv")
    print("✅ 已保存 rest_lengths.csv")


if __name__ == "__main__":
    run()
