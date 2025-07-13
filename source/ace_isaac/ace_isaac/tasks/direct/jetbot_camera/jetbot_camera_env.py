# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
#from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from .jetbot_camera_env_cfg import JetbotCameraEnvCfg

class JetbotCameraEnv(DirectRLEnv):
    cfg: JetbotCameraEnvCfg

    def __init__(self, cfg: JetbotCameraEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale
        self.robot        = self.scene["jetbot"]
        self.robot_camera = self.scene["camera"]
        self.goal_marker  = self.scene["goal_marker"]
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        print(self.dof_idx)

    def _set_goal_position(self):
        print("SET GOAL FUNCTION WAS CALLED")
        robot_orientation = self.robot.data.root_quat_w
        marker = self.scene["goal_marker"]
        # forward_vector = get_basis_vector_z(robot_orientation)
        positions, orientations = marker.get_world_poses()
        positions[:, 2] += 1.5
        marker.set_world_poses(positions, orientations) 
        forward_distance = 1
        # point_in_front = self.robot.data.root_pow_w + forward_distance * forward_vector
        return

    def _configure_gym_env_spaces(self):
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations

        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.scene.camera.height, self.cfg.scene.camera.width, self.cfg.num_channels),
        )
        self.single_action_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_actions,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # RL specifics
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale*actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions)

    def _get_observations(self) -> dict:
        observations =  self.robot_camera.data.output["rgb"].clone()
        # get rid of the alpha channel
        observations = observations[:, :, :, :3]
        return {"policy": observations}
    
    # def _get_rewards(self) -> torch.Tensor:
    #     robot_position = self.robot.data.root_pos_w
    #     goal_position = self.goal_marker.data.root_pos_w
    #     squared_diffs = (robot_position - goal_position) ** 2
    #     distance_to_goal = torch.sqrt(torch.sum(squared_diffs, dim=-1))
    #     rewards = torch.exp(1/(distance_to_goal))
    #     #rewards -= 4

    #     if (self.common_step_counter % 10 == 0):
    #         print(f"Reward at step {self.common_step_counter} is {rewards} for distance {distance_to_goal}")
    #     return rewards


    def _get_rewards(self) -> torch.Tensor:
        # existing distance‐based reward
        robot_position = self.robot.data.root_pos_w
        goal_position  = self.goal_marker.data.root_pos_w
        squared_diffs  = (robot_position - goal_position) ** 2
        distance_to_goal = torch.sqrt(torch.sum(squared_diffs, dim=-1))
        rewards = torch.exp(1.0 / (distance_to_goal + 1e-6))
 
        # --- encourage any forward/backward motion ---
        # root_com_lin_vel_w: (N,3) world‐frame COM velocity
        lin_vel = self.robot.data.root_com_lin_vel_w            # shape [num_envs,3]
        speed   = torch.linalg.norm(lin_vel, dim=-1)           # shape [num_envs]
        move_bonus = 0.05 * speed                               # tune 0.05
        rewards    = rewards + move_bonus

        # --- penalize yaw spinning in place ---
        # root_com_ang_vel_w[...,2]: yaw rate around vertical axis
        # ang_vel     = self.robot.data.root_com_ang_vel_w[..., 2]  # shape [num_envs]
        # spin_penalty= 0.1 * torch.abs(ang_vel)                   # tune 0.1
        # rewards     = rewards - spin_penalty
 
        if (self.common_step_counter % 10 == 0):
            print(f"Reward at step {self.common_step_counter} is {rewards} for distance {distance_to_goal}")
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return False, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        default_goal_root_state = self.goal_marker.data.default_root_state[env_ids]
        default_goal_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.goal_marker.write_root_pose_to_sim(default_goal_root_state[:, :7], env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
