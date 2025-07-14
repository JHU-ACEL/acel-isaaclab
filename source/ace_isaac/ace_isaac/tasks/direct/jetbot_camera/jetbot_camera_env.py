# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import TiledCamera

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


from .jetbot_camera_env_cfg import JetbotCameraEnvCfg

class JetbotCameraEnv(DirectRLEnv):
    cfg: JetbotCameraEnvCfg

    def __init__(self, cfg: JetbotCameraEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.goal_marker = RigidObject(self.cfg.goal_cfg)
        self.robot_camera = TiledCamera(self.cfg.tiled_camera)

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

        self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()  
        self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()
        self.commands = torch.randn((self.cfg.scene.num_envs, 3)).cuda()
        self.commands[:,-1] = 0.0
        self.commands = self.commands/torch.linalg.norm(self.commands, dim=1, keepdim=True)
        
        # offsets to account for atan range and keep things on [-pi, pi]
        ratio = self.commands[:,1]/(self.commands[:,0]+1E-8)
        gzero = torch.where(self.commands > 0, True, False)
        lzero = torch.where(self.commands < 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset[:,-1] = 0.5
        self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
        self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()

        self.radius_l = 0.5
        self.radius_h = 1.0
        self.dirs = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        

    def _visualize_markers(self):

        root_pos = self.robot.data.root_pos_w                # (N,3)
        goal_vec = self.goal_marker.data.root_pos_w - root_pos          # (N,3)

        #print(f"Position of the Red Block: {self.goal_marker.data.root_pos_w }")

        goal_vec = goal_vec/torch.linalg.norm(goal_vec, dim=-1, keepdim=True)
        self.commands = goal_vec
        ratio = self.commands[:,1]/(self.commands[:,0]+1E-8)
        gzero = torch.where(self.commands > 0, True, False)
        lzero = torch.where(self.commands < 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

        self.marker_locations = self.robot.data.root_pos_w
        self.forward_marker_orientations = self.robot.data.root_quat_w
        self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()

        loc = self.marker_locations + self.marker_offset
        loc = torch.vstack((loc, loc))
        rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))

        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))

        self.visualization_markers.visualize(loc, rots, marker_indices=indices)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = 10*actions.clone()# + torch.ones_like(actions)
        self._visualize_markers()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:

        camera_data = self.robot_camera.data.output["rgb"] / 255.0
        # normalize the camera data for better training results
        mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
        camera_data -= mean_tensor

        return {"policy": camera_data.clone()}

    def _get_rewards(self) -> torch.Tensor:
        forward_reward = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)

        forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)
        alignment_reward = torch.sum(forwards * self.commands, dim=-1, keepdim=True)
        
        # total_reward = forward_reward*alignment_reward
        # total_reward = forward_reward*alignment_reward + forward_reward
        # total_reward = forward_reward*torch.exp(alignment_reward)

        # arrival bonus and distance penalty
        root_pos = self.robot.data.root_pos_w                # (N,3)
        goal_vec = self.goal_marker.data.root_pos_w - root_pos          # (N,3)
        dist = torch.linalg.norm(goal_vec, dim=-1, keepdim=True)  # (N,1)
        arrived = (dist < 0.1).to(torch.float32) * 2.0
        dist_penalty = -0.1 * dist

        total_reward = forward_reward + alignment_reward + arrived + dist_penalty

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

        # default_block_root_state = self.goal_marker.data.default_root_state[env_ids]
        # default_block_root_state[:, :3] += self.scene.env_origins[env_ids]

        # self.goal_marker.write_root_state_to_sim(default_block_root_state, env_ids)

        N = self.cfg.scene.num_envs
        self.dirs = torch.randn((N, 3)).cuda()
        self.dirs[:,2] = 0.0
        self.dirs = self.dirs/torch.linalg.norm(self.dirs, dim=1, keepdim=True)

        goal_marker_pos = self.radius_l + (self.radius_h - self.radius_l) * torch.rand((N, 1)).cuda()
        goal_marker_pos = goal_marker_pos*self.dirs + self.robot.data.root_pos_w
        ### USE THIS TO SET THE POSITION OF THE GOAL MARKER###

        new_goal_state = self.goal_marker.data.default_root_state[env_ids].clone()
        new_goal_state[:, :3] = goal_marker_pos
        self.goal_marker.write_root_state_to_sim(new_goal_state, env_ids)

        root_pos = self.robot.data.root_pos_w                # (N,3)
        goal_vec = self.goal_marker.data.root_pos_w - root_pos          # (N,3)
        goal_vec = goal_vec/torch.linalg.norm(goal_vec, dim=-1, keepdim=True)
        self.commands[env_ids] = goal_vec

        ratio = self.commands[env_ids][:,1]/(self.commands[env_ids][:,0]+1E-8)
        gzero = torch.where(self.commands[env_ids] > 0, True, False)
        lzero = torch.where(self.commands[env_ids]< 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws[env_ids] = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

        self._visualize_markers()
