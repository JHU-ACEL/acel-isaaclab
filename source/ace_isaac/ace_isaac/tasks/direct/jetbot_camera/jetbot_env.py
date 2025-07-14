# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from .jetbot_env_cfg import JetbotEnvCfg


from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "goal": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.25, 0.25, 0.25),
                )
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)

class JetbotEnv(DirectRLEnv):
    cfg: JetbotEnvCfg

    def __init__(self, cfg: JetbotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        print(self.dof_idx)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.visualization_markers = define_markers()
        self.radius_l = 1.0
        self.radius_h = 3.0
        self.dirs = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()

    def _visualize_markers(self):
        loc = self.marker_locations
        loc = torch.vstack((loc, loc))
        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
        self.visualization_markers.visualize(loc, marker_indices=indices)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = 25*actions.clone()
        self._visualize_markers()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        # 1. world‐frame root position & orientation
        root_pos_w = self.robot.data.root_pos_w        # (N,3)
        root_rot_w = self.robot.data.root_quat_w        # (N,4) quaternion

        # 2. compute vector from robot to its goal, in world-frame
        goal_vec_w = self.marker_locations - root_pos_w  # (N,3)

        # 3. rotate into robot body frame
        goal_vec_b = math_utils.quat_apply_inverse(root_rot_w, goal_vec_w)  # (N,3)
        # math_utils.quat_apply_inverse() takes in a quaternion and some vector defined in some frame, and rotates this vector
        # by the inverse of the quaternion, so the vector is defined in another coordinate frame. 
        # originally, root_rot_w is the rotation quaternion of the robot coordinate frame with respect to world coordinate frame. 
        # Getting the inverse of this transforms any vector originally defined in the world coordinate frame, to the robot coordinate frame.
        # So we will get the vector from the robot to the goal defined wrt the robot coordinate frame instead of world coordinate frame. 

        # 4. get base velocity in body frame (already what you had)
        vel_b = self.robot.data.root_com_lin_vel_b      # (N,3)

        self.bodyvel = self.robot.data.root_com_lin_vel_b
        self.goal_vec = goal_vec_b
        self.goal_dist = torch.linalg.norm(goal_vec_b, dim=-1, keepdim=True)
        obs = torch.hstack((self.bodyvel, self.goal_vec, self.goal_dist))

        # 5. pack into dict (you can flatten or keep as vectors)
        observations = {"policy": obs}
        return observations


    def _get_rewards(self) -> torch.Tensor:
        # current root world pos & goal
        root_pos = self.robot.data.root_pos_w                # (N,3)
        goal_vec = self.marker_locations - root_pos          # (N,3)

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

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return False, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_state_to_sim(default_root_state, env_ids)

        N = self.cfg.scene.num_envs
        self.dirs = torch.randn((N, 3)).cuda()
        self.dirs[:,2] = 0.0
        self.dirs = self.dirs/torch.linalg.norm(self.dirs, dim=1, keepdim=True)
        self.marker_locations = self.radius_l + (self.radius_h - self.radius_l) * torch.rand((N, 1)).cuda()
        self.marker_locations = self.marker_locations*self.dirs + self.robot.data.root_pos_w