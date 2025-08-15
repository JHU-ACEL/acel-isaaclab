# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import TiledCamera
import isaaclab.utils.math as math_utils

from .jackal_terrain_env_cfg import JackalTerrainEnvCfg
from .terrain_utils.terrain_utils import TerrainManager

def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


class JackalTerrainEnv(DirectRLEnv):
    cfg: JackalTerrainEnvCfg

    def __init__(self, cfg: JackalTerrainEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        #spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(10000.0,10000.0)))
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.gpu = "cuda:0"

        # Arrow Markers Initialization (For Debugging/Visualization Purposes)
        self.arrows = define_markers()
        self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()  
        self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()
        self.commands = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()

        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3), device=self.gpu)
        self.marker_offset[:, 2] = 0.5

        self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4), device=self.gpu)
        self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4), device=self.gpu)
        self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()  
        self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()

        # Terrain Manager
        self.terrainManager = TerrainManager(
            num_envs=self.cfg.scene.num_envs, 
            device=self.gpu,
        )
        self.valid_spawns = self.terrainManager.spawn_locations

    def _update_marker_yaws(self):

        ratio = self.commands[:,1]/(self.commands[:,0]+1E-8)
        gzero = torch.where(self.commands > 0, True, False)
        lzero = torch.where(self.commands < 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

    def _visualize_markers(self):
        
        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))

        self._update_marker_yaws()
        self.marker_locations = self.robot.data.root_pos_w
        self.forward_marker_orientations = self.robot.data.root_quat_w
        self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()
        loc = self.marker_locations + self.marker_offset
        loc = torch.vstack((loc, loc))
        rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))

        self.arrows.visualize(loc, rots, marker_indices=indices)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._visualize_markers()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        #self.velocity = self.robot.data.root_com_vel_w
        self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)
        self.forwards[:,2] = 0.0
        self.forwards = self.forwards/torch.linalg.norm(self.forwards, dim=1, keepdim=True)

        dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        cross = torch.cross(self.forwards, self.commands, dim=-1)[:,-1].reshape(-1,1)
        forward_speed = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        obs = torch.hstack((dot, cross, forward_speed))

        observations = {"policy": obs}
        return observations
    
    def _get_rewards(self) -> torch.Tensor:
        forward_reward = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        alignment_reward = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        #total_reward = forward_reward + alignment_reward
        # total_reward = forward_reward*alignment_reward
        # total_reward = forward_reward*alignment_reward + forward_reward
        total_reward = forward_reward+2*torch.exp(alignment_reward)
        return total_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return False, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] = self.valid_spawns[env_ids]
        default_root_state[:, 2] += 0.075
        self.robot.write_root_state_to_sim(default_root_state, env_ids)

        self.commands[env_ids] = torch.randn((len(env_ids), 3)).cuda()
        self.commands[env_ids,-1] = 0.0
        self.commands[env_ids] = self.commands[env_ids]/torch.linalg.norm(self.commands[env_ids], dim=1, keepdim=True)
        self._visualize_markers()
