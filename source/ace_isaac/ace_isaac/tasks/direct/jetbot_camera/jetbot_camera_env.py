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

        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        print(self.dof_idx)

        self.prev_dist = torch.linalg.norm(self.robot.data.root_pos_w - self.goal_marker.data.root_pos_w, dim=-1)

    def _setup_scene(self):

        self.robot        = self.scene["jetbot"]
        self.robot_camera = self.scene["camera"]
        self.goal_marker  = self.scene["goal_marker"]

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale*actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:

        camera_data = self.robot_camera.data.output["rgb"] / 255.0
        # normalize the camera data for better training results
        mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
        camera_data -= mean_tensor

        return {"policy": camera_data.clone()}

    def _get_rewards(self):
        
        # current root world pos & goal
        root_pos = self.robot.data.root_pos_w                # (N,3)
        goal_vec = self.goal_marker.data.root_pos_w - root_pos          # (N,3)

        # distance to goal
        dist = torch.linalg.norm(goal_vec, dim=-1, keepdim=True)  # (N,1)

        # unit direction toward goal (avoid div by zero)
        dir_to_goal = goal_vec / (dist + 1e-6)

        # get COM velocity in world frame
        vel_w = self.robot.data.root_com_lin_vel_w         # (N,3)

        # 1) progress: projection of velocity onto goal direction
        progress = torch.sum(vel_w * dir_to_goal, dim=-1, keepdim=True)

        # 2) distance penalty (scaled negative)
        dist_penalty = -0.2 * dist

        # 3) sparse “arrival” bonus
        arrived = (dist < 0.2).to(torch.float32) * 2.0      # +2 when within 0.2 m

        # combine
        total_reward = progress + dist_penalty + arrived
        total_reward = total_reward.flatten()

        if (self.common_step_counter % 10 == 0):
            #print(f"Velocity Contribution is {progress}")
            print(f"Reward at step {self.common_step_counter} is {total_reward} with shape {total_reward.shape}")

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return False, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        default_goal_root_state = self.goal_marker.data.default_root_state[env_ids]
        default_goal_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.goal_marker.write_root_state_to_sim(default_goal_root_state, env_ids)

        # set the root state for the reset envs
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(default_root_state, env_ids)

        self.prev_dist = torch.linalg.norm(self.robot.data.root_pos_w - self.goal_marker.data.root_pos_w, dim=-1)
