import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
import os
from scipy.spatial.transform import Rotation
from collections import deque
import mujoco

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class tr_env_gym(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array"
        ],
        "render_fps": 50,
    }

    def __init__(
        self,
        xml_file=os.path.join(os.getcwd(),"t.xml"),
        use_contact_forces=False,
        use_tendon_length=False,
        use_cap_velocity=True,
        use_obs_noise=True,
        use_inherent_params_dr=True,
        terminate_when_unhealthy=True,
        is_test = False,
        desired_action = "straight",
        desired_direction = 1,
        ctrl_cost_weight=0.01,
        contact_cost_weight=5e-4,
        healthy_reward=0.1, 
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.0, # reset noise is handled in the following 4 variables
        min_reset_heading = 0.0,
        max_reset_heading = 2*np.pi,
        tendon_reset_mean = 0.0, # 0.15,
        tendon_reset_stdev = 0.0, # 0.2
        tendon_max_length = 0.05, # 0.15,
        tendon_min_length = -0.05, # -0.45,
        reward_delay_seconds = 0.02, # 0.5,
        # friction_noise_range = (0.25, 2.0),
        # damping_noise_range_side = (0.25, 4.0),
        # damping_noise_range_cross = (2.5, 40),
        # stiffness_noise_range_side = (5, 20),
        # stiffness_noise_range_cross = (75, 300),
        friction_noise_range = (1, 1),
        damping_noise_range_side = (1, 1),
        damping_noise_range_cross = (10, 10),
        stiffness_noise_range_side = (10, 10),
        stiffness_noise_range_cross = (150, 150),
        contact_with_self_penalty = 0.0,
        obs_noise_tendon_stdev = 0.02,
        obs_noise_cap_pos_stdev = 0.01,
        way_pts_range = (2.5, 3.5),
        way_pts_angle_range = (-np.pi/6, np.pi/6),
        threshold_waypt = 0.05,
        ditch_reward_max=300,
        ditch_reward_stdev=0.15,
        waypt_reward_amplitude=100,
        waypt_reward_stdev=0.10,
        yaw_reward_weight=1,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            use_contact_forces,
            use_tendon_length,
            use_cap_velocity,
            use_obs_noise,
            use_inherent_params_dr,
            terminate_when_unhealthy,
            is_test,
            desired_action,
            desired_direction,
            ctrl_cost_weight,
            contact_cost_weight,
            healthy_reward,
            contact_force_range,
            reset_noise_scale,
            min_reset_heading,
            max_reset_heading,
            tendon_reset_mean,
            tendon_reset_stdev,
            tendon_max_length,
            tendon_min_length,
            reward_delay_seconds,
            friction_noise_range,
            damping_noise_range_side,
            damping_noise_range_cross,
            stiffness_noise_range_side,
            stiffness_noise_range_cross,
            contact_with_self_penalty,
            obs_noise_tendon_stdev,
            obs_noise_cap_pos_stdev,
            way_pts_range,
            way_pts_angle_range,
            threshold_waypt,
            ditch_reward_max,
            ditch_reward_stdev,
            waypt_reward_amplitude,
            waypt_reward_stdev,
            yaw_reward_weight,
            **kwargs
        )
        self._x_velocity = 1
        self._y_velocity = 1
        self._is_test = is_test
        self._desired_action = desired_action
        self._desired_direction = desired_direction
        self._reset_psi = 0
        self._psi_wrap_around_count = 0
        self._use_tendon_length = use_tendon_length
        self._use_cap_velocity = use_cap_velocity
        
        self._oripoint = np.array([0.0, 0.0])
        self._waypt_range = way_pts_range
        self._waypt_angle_range = way_pts_angle_range
        self._threshold_waypt = threshold_waypt
        self._ditch_reward_max = ditch_reward_max
        self._ditch_reward_stdev = ditch_reward_stdev
        self._waypt_reward_amplitude = waypt_reward_amplitude
        self._waypt_reward_stdev = waypt_reward_stdev
        self._yaw_reward_weight = yaw_reward_weight
        self._waypt = np.array([])

        self._lin_vel_cmd = np.array([0.0, 0.0])
        self._ang_vel_cmd = 0.0


        self._use_obs_noise = use_obs_noise
        self._obs_noise_tendon_stdev = obs_noise_tendon_stdev
        self._obs_noise_cap_pos_stdev = obs_noise_cap_pos_stdev
        self._use_inherent_params_dr = use_inherent_params_dr

        self._min_reset_heading = min_reset_heading
        self._max_reset_heading = max_reset_heading
        self._tendon_reset_mean = tendon_reset_mean
        self._tendon_reset_stdev = tendon_reset_stdev
        self._tendon_max_length = tendon_max_length
        self._tendon_min_length = tendon_min_length

        self._friction_noise_range = friction_noise_range
        self._damping_noise_range_side = damping_noise_range_side
        self._damping_noise_range_cross = damping_noise_range_cross
        self._stiffness_noise_range_side = stiffness_noise_range_side
        self._stiffness_noise_range_cross = stiffness_noise_range_cross

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_force_range = contact_force_range
        if self._desired_action == "turn":
            self._contact_force_range = (-1000.0, 1000.0)
        self._reset_noise_scale = reset_noise_scale
        self._use_contact_forces = use_contact_forces

        self._contact_with_self_penalty = contact_with_self_penalty

        obs_shape = 18
        if use_tendon_length:
            obs_shape += 9
        if use_contact_forces:
            obs_shape += 84
        if use_cap_velocity:
            obs_shape += 18
        if desired_action == "tracking" or desired_action == "aiming" or desired_action == "vel_track":
            obs_shape += 3 # cmd lin_vel * 2 + ang_vel * 1
        
        self.state_shape = obs_shape

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )
        frame_skip = 20
        MujocoEnv.__init__(
            self, xml_file, frame_skip, observation_space=observation_space, **kwargs
        )
        self._reward_delay_steps = int(reward_delay_seconds/self.dt)
        self._heading_buffer = deque()


    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()

        min_velocity = 0.0001
        is_healthy = np.isfinite(state).all() and ((self._x_velocity > min_velocity or self._x_velocity < -min_velocity) \
                                                        or (self._y_velocity > min_velocity or self._y_velocity < -min_velocity) )
            
        
        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    def step(self, action):
        
        xy_position_before = (self.get_body_com("r01_body")[:2].copy() + \
                            self.get_body_com("r23_body")[:2].copy() + \
                            self.get_body_com("r45_body")[:2].copy())/3
        
        pos_r01_left_end = self.data.geom("s0").xpos.copy()
        pos_r23_left_end = self.data.geom("s2").xpos.copy()
        pos_r45_left_end = self.data.geom("s4").xpos.copy()
        left_COM_before = (pos_r01_left_end+pos_r23_left_end+pos_r45_left_end)/3
        pos_r01_right_end = self.data.geom("s1").xpos.copy()
        pos_r23_right_end = self.data.geom("s3").xpos.copy()
        pos_r45_right_end = self.data.geom("s5").xpos.copy()
        right_COM_before = (pos_r01_right_end+pos_r23_right_end+pos_r45_right_end)/3

        orientation_vector_before = left_COM_before - right_COM_before
        psi_before = np.arctan2(-orientation_vector_before[0], orientation_vector_before[1])
        filtered_action = self._action_filter(action, self.data.ctrl[:].copy())
        print("filtered_action:",filtered_action)
        
        self.do_simulation(filtered_action, self.frame_skip)
        xy_position_after = (self.get_body_com("r01_body")[:2].copy() + \
                            self.get_body_com("r23_body")[:2].copy() + \
                            self.get_body_com("r45_body")[:2].copy())/3

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        self._x_velocity, self._y_velocity = xy_velocity

        x_position_before, y_position_before = xy_position_before
        x_position_after, y_position_after = xy_position_after

        pos_r01_left_end = self.data.geom("s0").xpos.copy()
        pos_r23_left_end = self.data.geom("s2").xpos.copy()
        pos_r45_left_end = self.data.geom("s4").xpos.copy()
        left_COM_after = (pos_r01_left_end+pos_r23_left_end+pos_r45_left_end)/3
        pos_r01_right_end = self.data.geom("s1").xpos.copy()
        pos_r23_right_end = self.data.geom("s3").xpos.copy()
        pos_r45_right_end = self.data.geom("s5").xpos.copy()
        right_COM_after = (pos_r01_right_end+pos_r23_right_end+pos_r45_right_end)/3

        orientation_vector_after = left_COM_after - right_COM_after
        psi_after = np.arctan2(-orientation_vector_after[0], orientation_vector_after[1])

        tendon_length = np.array(self.data.ten_length)
        tendon_length_6 = tendon_length[:6]
        print("state:",tendon_length)
        state, observation,global_obs = self._get_obs()
        done = state[0]==np.nan
        #print("state", state)
        return observation,done,state

    def _get_obs(self):
        
        
        """ rotation_r01 = Rotation.from_matrix(self.data.geom("r01").xmat.reshape(3,3)).as_quat() # 4
        rotation_r23 = Rotation.from_matrix(self.data.geom("r23").xmat.reshape(3,3)).as_quat() # 4
        rotation_r45 = Rotation.from_matrix(self.data.geom("r45").xmat.reshape(3,3)).as_quat() # 4 """

        pos_r01_left_end = self.data.geom("s0").xpos.copy()
        pos_r01_right_end = self.data.geom("s1").xpos.copy()
        pos_r23_left_end = self.data.geom("s2").xpos.copy()
        pos_r23_right_end = self.data.geom("s3").xpos.copy()
        pos_r45_left_end = self.data.geom("s4").xpos.copy()
        pos_r45_right_end = self.data.geom("s5").xpos.copy()

        pos_center = (pos_r01_left_end + pos_r01_right_end + pos_r23_left_end + pos_r23_right_end + pos_r45_left_end + pos_r45_right_end) / 6

        pos_rel_s0 = pos_r01_left_end - pos_center # 3
        pos_rel_s1 = pos_r01_right_end - pos_center # 3
        pos_rel_s2 = pos_r23_left_end - pos_center # 3
        pos_rel_s3 = pos_r23_right_end - pos_center # 3
        pos_rel_s4 = pos_r45_left_end - pos_center # 3
        pos_rel_s5 = pos_r45_right_end - pos_center # 3

        rng = np.random.default_rng()
        random = rng.standard_normal(size=3)
        pos_rel_s0_with_noise = random * self._obs_noise_cap_pos_stdev + pos_rel_s0 # 3
        random = rng.standard_normal(size=3)
        pos_rel_s1_with_noise = random * self._obs_noise_cap_pos_stdev + pos_rel_s1 # 3
        random = rng.standard_normal(size=3)
        pos_rel_s2_with_noise = random * self._obs_noise_cap_pos_stdev + pos_rel_s2 # 3
        random = rng.standard_normal(size=3)
        pos_rel_s3_with_noise = random * self._obs_noise_cap_pos_stdev + pos_rel_s3 # 3
        random = rng.standard_normal(size=3)
        pos_rel_s4_with_noise = random * self._obs_noise_cap_pos_stdev + pos_rel_s4 # 3
        random = rng.standard_normal(size=3)
        pos_rel_s5_with_noise = random * self._obs_noise_cap_pos_stdev + pos_rel_s5 # 3

        # do not include positional data in the observation
        # position_r01 = self.data.geom("r01").xvelp
        # position_r23 = self.data.geom("r23").xvelp
        # position_r45 = self.data.geom("r45").xvelp
        global_state = np.concatenate((pos_r01_left_end, pos_r01_right_end, pos_r23_left_end, pos_r23_right_end, pos_r45_left_end, pos_r45_right_end))
        tendon_lengths = self.data.ten_length[-9:] # 9
        
        random = rng.standard_normal(size=9)
        tendon_lengths_with_noise = random * self._obs_noise_tendon_stdev + tendon_lengths # 9

        state = np.concatenate((pos_rel_s0,pos_rel_s1,pos_rel_s2, pos_rel_s3, pos_rel_s4, pos_rel_s5))
        state_with_noise = np.concatenate((pos_rel_s0_with_noise, pos_rel_s1_with_noise, pos_rel_s2_with_noise, pos_rel_s3_with_noise, pos_rel_s4_with_noise, pos_rel_s5_with_noise))
        
        if self._use_obs_noise == True:
            observation = state_with_noise
        else:
            observation = state

        if self._use_cap_velocity:
            velocity = self.data.qvel # 18

            vel_lin_r01 = np.array([velocity[0], velocity[1], velocity[2]])
            vel_ang_r01 = np.array([velocity[3], velocity[4], velocity[5]])
            vel_lin_r23 = np.array([velocity[6], velocity[7], velocity[8]])
            vel_ang_r23 = np.array([velocity[9], velocity[10], velocity[11]])
            vel_lin_r45 = np.array([velocity[12], velocity[13], velocity[14]])
            vel_ang_r45 = np.array([velocity[15], velocity[16], velocity[17]])

            s0_r01_pos = pos_r01_left_end - self.data.body("r01_body").xpos.copy()
            s1_r01_pos = pos_r01_right_end - self.data.body("r01_body").xpos.copy()
            s2_r23_pos = pos_r23_left_end - self.data.body("r23_body").xpos.copy()
            s3_r23_pos = pos_r23_right_end - self.data.body("r23_body").xpos.copy()
            s4_r45_pos = pos_r45_left_end - self.data.body("r45_body").xpos.copy()
            s5_r45_pos = pos_r45_right_end - self.data.body("r45_body").xpos.copy()

            vel_s0 = vel_lin_r01 + np.cross(vel_ang_r01, s0_r01_pos) # 3
            vel_s1 = vel_lin_r01 + np.cross(vel_ang_r01, s1_r01_pos) # 3
            vel_s2 = vel_lin_r23 + np.cross(vel_ang_r23, s2_r23_pos) # 3
            vel_s3 = vel_lin_r23 + np.cross(vel_ang_r23, s3_r23_pos) # 3
            vel_s4 = vel_lin_r45 + np.cross(vel_ang_r45, s4_r45_pos) # 3
            vel_s5 = vel_lin_r45 + np.cross(vel_ang_r45, s5_r45_pos) # 3

            random = rng.standard_normal(size=3)
            vel_s0_with_noise = random * self._obs_noise_cap_pos_stdev + vel_s0 # 3
            random = rng.standard_normal(size=3)
            vel_s1_with_noise = random * self._obs_noise_cap_pos_stdev + vel_s1 # 3
            random = rng.standard_normal(size=3)
            vel_s2_with_noise = random * self._obs_noise_cap_pos_stdev + vel_s2 # 3
            random = rng.standard_normal(size=3)
            vel_s3_with_noise = random * self._obs_noise_cap_pos_stdev + vel_s3 # 3
            random = rng.standard_normal(size=3)
            vel_s4_with_noise = random * self._obs_noise_cap_pos_stdev + vel_s4 # 3
            random = rng.standard_normal(size=3)
            vel_s5_with_noise = random * self._obs_noise_cap_pos_stdev + vel_s5 # 3

            state = np.concatenate((pos_rel_s0,pos_rel_s1,pos_rel_s2, pos_rel_s3, pos_rel_s4, pos_rel_s5,\
                                        vel_s0, vel_s1, vel_s2, vel_s3, vel_s4, vel_s5))
            state_with_noise = np.concatenate((pos_rel_s0_with_noise, pos_rel_s1_with_noise, pos_rel_s2_with_noise, pos_rel_s3_with_noise, pos_rel_s4_with_noise, pos_rel_s5_with_noise,\
                                        vel_s0_with_noise, vel_s1_with_noise, vel_s2_with_noise, vel_s3_with_noise, vel_s4_with_noise, vel_s5_with_noise))
            
        if self._use_tendon_length:
            state = np.concatenate((state, tendon_lengths))
            state_with_noise = np.concatenate((state_with_noise, tendon_lengths_with_noise))

        if self._desired_action == "tracking" or self._desired_action == "aiming":
            tracking_vec = self._waypt - pos_center[:2]
            tgt_drct = tracking_vec / np.linalg.norm(tracking_vec)
            pos_center_noise_del = (pos_rel_s0_with_noise + pos_rel_s1_with_noise + pos_rel_s2_with_noise + pos_rel_s3_with_noise + pos_rel_s4_with_noise + pos_rel_s5_with_noise)/6
            tracking_vec_with_noise = tracking_vec - pos_center_noise_del[:2]
            tgt_drct_with_noise = tracking_vec_with_noise / np.linalg.norm(tracking_vec_with_noise)

            tgt_yaw = np.array([np.arctan2(tgt_drct[1], tgt_drct[0])])
            tgt_yaw_with_noise = np.array([np.arctan2(tgt_drct_with_noise[1], tgt_drct_with_noise[0])])

            state = np.concatenate((state,\
                                          tracking_vec, tgt_yaw))
            state_with_noise = np.concatenate((state_with_noise,\
                                                     tracking_vec_with_noise, tgt_yaw_with_noise))
        
        if self._desired_action == "vel_track":
            vel_cmd = np.array([self._lin_vel_cmd[0], self._lin_vel_cmd[1], self._ang_vel_cmd])
            state = np.concatenate((state, vel_cmd))
            state_with_noise = np.concatenate((state_with_noise, vel_cmd))

        return state, observation, global_state

    def _angle_normalize(self, theta):
        if theta > np.pi:
            return self._angle_normalize(theta - 2 * np.pi)
        elif theta <= -np.pi:
            return self._angle_normalize(theta + 2 * np.pi)
        else:
            return theta
    
    def _ditch_reward(self, xy_position):
        pointing_vec = self._waypt - self._oripoint
        dist_pointing = np.linalg.norm(pointing_vec)
        pointing_vec_norm = pointing_vec / dist_pointing

        tracking_vec = self._waypt - xy_position
        dist_along = np.dot(tracking_vec, pointing_vec_norm)
        dist_bias = np.linalg.norm(tracking_vec - dist_along*pointing_vec_norm)

        ditch_rew = self._ditch_reward_max * (1.0 - np.abs(dist_along)/dist_pointing) * np.exp(-dist_bias**2 / (2*self._ditch_reward_stdev**2))
        waypt_rew = self._waypt_reward_amplitude * np.exp(-np.linalg.norm(xy_position - self._waypt)**2 / (2*self._waypt_reward_stdev**2))
        return ditch_rew+waypt_rew
    
    def _vel_track_rew(self, vel_cmd, vel_bwd):
        track_stdev = np.array([5.0, 7.0])
        track_amplitude = np.array([1.0, 0.5])
        lin_vel_err = np.linalg.norm(vel_bwd[0:2] - vel_cmd[0:2])
        ang_vel_err = vel_bwd[2] - vel_cmd[2]

        lin_track_rew = track_amplitude[0] * np.exp(-track_stdev[0] * lin_vel_err**2)
        ang_track_rew = track_amplitude[1] * np.exp(-track_stdev[1] * ang_vel_err**2)

        return lin_track_rew + ang_track_rew
    
    def _action_filter(self, action, last_action):
        k_FILTER = 5
        vel_constraint = 0.1

        # del_action = np.clip(k_FILTER*(action - last_action)*self.dt, -vel_constraint*self.dt, vel_constraint*self.dt)
        # del_action = k_FILTER*(action - last_action)*self.dt
        del_action = action / 0.05 * vel_constraint*self.dt

        filtered_action = last_action + del_action
        #return filtered_action
        return action

    def _reset_inherent_params(self):
        friction_coeff = np.random.uniform(self._friction_noise_range[0], self._friction_noise_range[1])
        damping_coeff = np.array([np.random.uniform(self._damping_noise_range_side[0], self._damping_noise_range_side[1]), np.random.uniform(self._damping_noise_range_cross[0], self._damping_noise_range_cross[1])])
        stiffness_coeff = np.array([np.random.uniform(self._stiffness_noise_range_side[0], self._stiffness_noise_range_side[1]), np.random.uniform(self._stiffness_noise_range_cross[0], self._stiffness_noise_range_cross[1])])

        self.model.geom_friction[:, 0] = friction_coeff
        self.model.tendon_damping[6:12] = damping_coeff[0]
        self.model.tendon_damping[12:15] = damping_coeff[1]
        self.model.tendon_stiffness[6:12] = stiffness_coeff[0]
        self.model.tendon_stiffness[12:15] = stiffness_coeff[1]
        return


    def reset_model(self):
        self._psi_wrap_around_count = 0

        if self._use_inherent_params_dr:
            self._reset_inherent_params()

        # '''
        # with rolling noise start
        rolling_qpos = [[0.2438013,  -0.23055046,  0.10995744,  0.46165276, -0.61078778, -0.64202933, 
                         -0.04016669,  0.23304155, -0.2781429,   0.0948906,   0.57252615,  0.17486495, 
                         -0.48006247, -0.64123013,  0.24824598, -0.2435365,   0.06010128,  0.12428316,  
                         0.77737256,  0.16439319, -0.59432355]]

        # idx_qpos = np.random.randint(0, 6)
        idx_qpos = 0
        qpos = rolling_qpos[idx_qpos]
        
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=len(qpos)
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)
        # with rolling noise end

        '''
        # without rolling noise start
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        if self._desired_action == "turn" or self._desired_action == "tracking" or self._desired_action == "aiming":
            self.set_state(qpos, qvel)
        # without rolling noise end
        #'''
        
        position_r01 = qpos[0:3]
        rotation_r01 = Rotation.from_quat([qpos[4], qpos[5], qpos[6], qpos[3]]).as_euler('xyz')
        position_r23 = qpos[7:10]
        rotation_r23 = Rotation.from_quat([qpos[11], qpos[12], qpos[13], qpos[10]]).as_euler('xyz')
        position_r45 = qpos[14:17]
        rotation_r45 = Rotation.from_quat([qpos[18], qpos[19], qpos[20], qpos[17]]).as_euler('xyz')

        ux = 0
        uy = 0
        uz = 1
        theta = np.random.uniform(low=self._min_reset_heading, high=self._max_reset_heading)
        # theta = -40 * np.pi / 180
        R = np.array([[np.cos(theta)+ux**2*(1-np.cos(theta)), 
                       ux*uy*(1-np.cos(theta))-uz*np.sin(theta),
                       ux*uz*(1-np.cos(theta))+uy*np.sin(theta)],
                       [uy*ux*(1-np.cos(theta))+uz*np.sin(theta),
                        np.cos(theta)+uy**2*(1-np.cos(theta)),
                        uy*uz*(1-np.cos(theta)-ux*np.sin(theta))],
                        [uz*ux*(1-np.cos(theta)) -uy*np.sin(theta),
                         uz*uy*(1-np.cos(theta)) + ux*np.sin(theta),
                         np.cos(theta)+uz**2*(1-np.cos(theta))]])

        
        position_r01_new = (R @ position_r01.reshape(-1,1)).squeeze()
        position_r23_new = (R @ position_r23.reshape(-1,1)).squeeze()
        position_r45_new = (R @ position_r45.reshape(-1,1)).squeeze()
        rot_quat_r01_new = Rotation.from_euler('xyz', rotation_r01 + [0, 0, theta]).as_quat()
        rot_quat_r01_new = [rot_quat_r01_new[3], rot_quat_r01_new[0], rot_quat_r01_new[1], rot_quat_r01_new[2]]
        rot_quat_r23_new = Rotation.from_euler('xyz', rotation_r23 + [0, 0, theta]).as_quat()
        rot_quat_r23_new = [rot_quat_r23_new[3], rot_quat_r23_new[0], rot_quat_r23_new[1], rot_quat_r23_new[2]]
        rot_quat_r45_new = Rotation.from_euler('xyz', rotation_r45 + [0, 0, theta]).as_quat()
        rot_quat_r45_new = [rot_quat_r45_new[3], rot_quat_r45_new[0], rot_quat_r45_new[1], rot_quat_r45_new[2]]

        qpos_new = np.concatenate((position_r01_new, rot_quat_r01_new, position_r23_new, rot_quat_r23_new,
                                   position_r45_new, rot_quat_r45_new))
        self.set_state(qpos_new, qvel)

        rng = np.random.default_rng()
        random = rng.standard_normal(size=6)
        tendons = random*self._tendon_reset_stdev + self._tendon_reset_mean
        for i in range(tendons.size):
            if tendons[i] > self._tendon_max_length:
                tendons[i] = self._tendon_max_length
            elif tendons[i] < self._tendon_min_length:
                tendons[i] = self._tendon_min_length
        
        for i in range(50):
            self.do_simulation(tendons, self.frame_skip)


        pos_r01_left_end = self.data.geom("s0").xpos.copy()
        pos_r23_left_end = self.data.geom("s2").xpos.copy()
        pos_r45_left_end = self.data.geom("s4").xpos.copy()
        left_COM_before = (pos_r01_left_end+pos_r23_left_end+pos_r45_left_end)/3
        pos_r01_right_end = self.data.geom("s1").xpos.copy()
        pos_r23_right_end = self.data.geom("s3").xpos.copy()
        pos_r45_right_end = self.data.geom("s5").xpos.copy()
        right_COM_before = (pos_r01_right_end+pos_r23_right_end+pos_r45_right_end)/3
        orientation_vector_before = left_COM_before - right_COM_before
        self._reset_psi = np.arctan2(-orientation_vector_before[0], orientation_vector_before[1])
                
        self._step_num = 0
        if self._desired_action == "turn" or self._desired_action == "aiming":
            for i in range(self._reward_delay_steps):
                self.step(tendons)
        state, _,observation = self._get_obs()

        return state, observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def get_rod_pairs(self):
        return [(0,1),(2,3),(4,5)]
    
    def get_spring_pairs(self):
        return [(1,4),(0,3),(2,5)]
    
    def get_cable_pairs(self):
        return [(0,4),(0,2),(2,4),(1,5),(3,5),(1,3)]
    
    def get_rest_lengths(self):
        return self.data.ten_length[0:9]+0.1312
    
    def get_stiffnesses(self):
        return [[700,0.8],[700,0.8],[700,0.8]]
    
    def get_rod_masses(self):
        return [3,3,3]
    
    def get_fixed_nodes(self):
        def find_closest_indices(lst):
            min_three_indices = sorted(range(len(lst)), key=lambda i: lst[i])[:3]
            min_three_indices.sort()  

            min_three_values = [lst[i] for i in min_three_indices]

            diff1 = abs(min_three_values[1] - min_three_values[0])
            diff2 = abs(min_three_values[2] - min_three_values[1])

            if diff1 < 1e-4 and diff2 < 1e-4:
                return min_three_indices
            else:
                return [-1, -1, -1]
        geoms = ["s0", "s1", "s2", "s3", "s4", "s5"]
        geom_height = [self.data.geom(geom).xpos[2].copy() for geom in geoms]
        print(geom_height)
        return find_closest_indices(geom_height)
        