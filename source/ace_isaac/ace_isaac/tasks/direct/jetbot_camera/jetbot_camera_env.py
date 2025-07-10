# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
#from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv
#from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from .jetbot_camera_env_cfg import JetbotCameraEnvCfg





# from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# import isaaclab.utils.math as math_utils

# def define_markers() -> VisualizationMarkers:
#     """Define markers with various different shapes."""
#     marker_cfg = VisualizationMarkersCfg(
#         prim_path="/Visuals/myMarkers",
#         markers={
#                 "goal": sim_utils.UsdFileCfg(
#                     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
#                     scale=(0.25, 0.25, 0.25),
#                 )
#         },
#     )
#     return VisualizationMarkers(cfg=marker_cfg)



class JetbotCameraEnv(DirectRLEnv):
    cfg: JetbotCameraEnvCfg

    def __init__(self, cfg: JetbotCameraEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        print(self.dof_idx)

    def _setup_scene(self):

        self.robot        = self.scene["jetbot"]
        self.robot_camera = self.scene["camera"]
        self.goal_marker  = self.scene["goal_marker"]

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


        # self.visualization_markers = define_markers()
        # self.radius_l = 0.7
        # self.radius_h = 0.7
        # self.dirs = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        # self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()

    # def _visualize_markers(self):
    #     loc = self.marker_locations
    #     loc = torch.vstack((loc, loc))
    #     all_envs = torch.arange(self.cfg.scene.num_envs)
    #     indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
    #     self.visualization_markers.visualize(loc, marker_indices=indices)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        #self._visualize_markers()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        observations =  self.robot_camera.data.output["rgb"].clone()
        # get rid of the alpha channel
        observations = observations[:, :, :, :3]
        return {"policy": observations}


    def _get_rewards(self) -> torch.Tensor:
        robot_position = self.robot.data.root_pos_w
        goal_position = self.goal_marker.data.root_pos_w
        squared_diffs = (robot_position - goal_position) ** 2
        distance_to_goal = torch.sqrt(torch.sum(squared_diffs, dim=-1))
        rewards = torch.exp(1/(distance_to_goal))
        rewards -= 30

        if (self.common_step_counter % 10 == 0):
            print(f"Reward at step {self.common_step_counter} is {rewards} for distance {distance_to_goal}")
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        epsilon = .01
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        robot_position = self.robot.data.root_pos_w
        goal_position = self.goal_marker.data.root_pos_w
        squared_diffs = (robot_position - goal_position) ** 2
        distance_to_goal = torch.sqrt(torch.sum(squared_diffs, dim=-1))
        distance_within_epsilon = distance_to_goal < epsilon
        distance_over_limit = distance_to_goal > .31
        position_termination_condition = torch.logical_or(distance_within_epsilon, distance_over_limit)
        position_termination_condition.fill_(False)
        return (position_termination_condition, time_out)

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
