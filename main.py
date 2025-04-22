import tr_env_gym
import os
import numpy as np
from get_action import *
from planning import COM_scheduler
import cv2
import os

def test_frame(env):
    frame =env.render()
    cv2.imwrite(f'frames/start.png', frame) 
    
    state, observation,global_obs=env._get_obs()
    print("observation:", global_obs)
    exit(1)

def run():
    # 1. 初始化环境与调度器
    env = tr_env_gym.tr_env_gym(
        render_mode="rgb_array",
        xml_file=os.path.join(os.getcwd(), "3prism_jonathan_steady_side.xml"),
        is_test=False,
        desired_action="straight",
        desired_direction=1,
        terminate_when_unhealthy=True
    )
    #test_frame(env)
    scheduler = COM_scheduler(1, 0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义编码器
    out = cv2.VideoWriter('output.mp4', fourcc, 60, (640, 480))
    os.makedirs('frames', exist_ok=True)
    # 2. 获取初始化数据（只能通过接口）
    env.render()
    state, observation,global_obs=env._get_obs()

    _,_,env_nodes = env._get_obs() # 获取节点位置
    env_nodes = np.array(env_nodes).reshape(-1,3)  # 仅包含节点位置
    rod_pairs = env.get_rod_pairs()
    elastic_cables = env.get_spring_pairs()
    rigid_cables = env.get_cable_pairs()
    rest_lengths = np.array(env.get_rest_lengths())  # 仅包含 rigid cables 的
    stiffness = np.array(env.get_stiffnesses())      # 对应 rigid cables
    mass = np.array(env.get_rod_masses())
    fixed_nodes = env.get_fixed_nodes()

    # 3. 初始化结构
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

    q_current = rest_lengths.copy()[:6]
    rest_lengths_history = []
    os.makedirs("figs", exist_ok=True)

    # 4. 控制主循环
    for step in range(100):
        update_position_from_env(structure, env)

        # 获取当前 COM
        com = structure.center_of_mass(np.array(structure.node_positions))
        x0, y0 = com[0], com[1]

        
        x1, y1 = structure.node_positions[0][:2]
        x2, y2 = structure.node_positions[3][:2]
        x3, y3 = structure.node_positions[4][:2]

        # 由调度器给出目标 COM
        (x_target, y_target), _ = scheduler.get_COM(x0, y0, x1, y1, x2, y2, x3, y3)
        target_com = np.array([x_target, y_target,com[2]])

        # 执行单步 IK 解算
        q_next, nodes = ik_step(structure, q_current, target_com, history=rest_lengths_history, step=step)

        action = q_next - q_current
        obs, done,  info = env.step(action[:6])
        frame =env.render()
        cv2.imwrite(f'frames/frame_{step:04d}.png', frame) 
        out.write(frame)
        out.release()

        q_current = q_next.copy()

        print(f"[step {step}] COM = {structure.center_of_mass(nodes)}")

        # 可视化结构保存
        fig_path = os.path.join("figs", f"step_{step:03d}.png")
        save_structure_plot(nodes, step, save_dir="figs")

        if done :
            print("❗️仿真终止")
            break

    # 5. 输出绳长历史
    save_rest_lengths_csv(rest_lengths_history, filename="rest_lengths.csv")
    print("✅ 已保存 rest_lengths.csv")

if __name__ == "__main__":
    run()