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
from .terrain_utils.usd_utils import get_triangles_and_vertices_from_prim

GOAL_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.CuboidCfg(
            size=(0.25, 0.25, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)

# Mesh Vertices Markers
MESH_MARKERS_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 1.0)),
        ),
    }
)

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


def sample_min_separation_xy(points_xyz: torch.Tensor, N: int, M: float, seed_idx: int | None = None):
    """
    points_xyz: (P, 3) float tensor
    N: number of points you want (will return <= N if impossible)
    M: minimum XY separation
    seed_idx: optional starting index for reproducibility

    Returns:
        idx (K,) long tensor of chosen indices (K <= N)
        pts (K, 3) tensor of chosen points
    """
    assert points_xyz.ndim == 2 and points_xyz.shape[1] == 3
    P = points_xyz.shape[0]
    device = points_xyz.device
    xy = points_xyz[:, :2].to(torch.float32)

    # Choose a seed
    if seed_idx is None:
        seed_idx = torch.randint(0, P, (1,), device=device).item()

    chosen = torch.empty(N, dtype=torch.long, device=device)
    chosen[0] = seed_idx

    # Track each point's distance^2 to the nearest chosen point so far
    # Initialize with distances to the seed
    diff = xy - xy[seed_idx]
    min_d2 = (diff * diff).sum(dim=1)  # (P,)

    M2 = float(M) * float(M)

    k = 1
    while k < N:
        # Pick the point that is farthest from the chosen set
        far_idx = torch.argmax(min_d2)

        # If even the farthest point is closer than M, we cannot add more
        if min_d2[far_idx] < M2:
            break

        chosen[k] = far_idx

        # Update min distances with the newly selected point
        diff = xy - xy[far_idx]
        d2 = (diff * diff).sum(dim=1)
        min_d2 = torch.minimum(min_d2, d2)

        k += 1

    chosen = chosen[:k]
    return chosen, points_xyz.index_select(0, chosen)


def trim_boundary_aabb(points_xyz: torch.Tensor, M: float):
    """
    Remove points within M of the axis-aligned XY boundary
    that encloses the terrain mesh.
    points_xyz: (P, 3) tensor
    """
    xy = points_xyz[:, :2]
    mins = xy.min(dim=0).values           # (2,)
    maxs = xy.max(dim=0).values           # (2,)
    mask = (
        (xy[:, 0] >= mins[0] + M) & (xy[:, 0] <= maxs[0] - M) &
        (xy[:, 1] >= mins[1] + M) & (xy[:, 1] <= maxs[1] - M)
    )
    keep_idx = mask.nonzero(as_tuple=False).squeeze(1)
    return keep_idx, points_xyz[keep_idx]


class JackalTerrainEnv(DirectRLEnv):
    cfg: JackalTerrainEnvCfg

    def __init__(self, cfg: JackalTerrainEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

    def _setup_scene(self):

        self.gpu = "cuda:0"
        self.max_timesteps = 12000

        self.robot = Articulation(self.cfg.robot_cfg)
        self.robot_camera = TiledCamera(self.cfg.tiled_camera)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["tiled_camera"] = self.robot_camera 

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

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

        # Goal Markers Initialization
        self.target_spawns = torch.zeros(self.cfg.scene.num_envs, 3, device=self.gpu)
        cube_cfg = GOAL_MARKER_CFG.copy()
        cube_cfg.prim_path = "/Visuals/Command/position_goal"
        self.goal_markers = VisualizationMarkers(cfg=cube_cfg)
        self.goal_markers.set_visibility(True)
        self.goal_radii = torch.empty(self.scene.num_envs, device=self.gpu, dtype=torch.float32)

        # Data structure to store observation history
        self.history_len = 5
        self._camera_hist: torch.Tensor | None = None

        # Get Mesh vertices and faces
        terrain_prim_path = "/World/terrain/terrain/ground"
        faces, self.mesh_vertices = get_triangles_and_vertices_from_prim(terrain_prim_path)
        self.mesh_vertices = torch.from_numpy(self.mesh_vertices).to(self.gpu)
        self.face_vertices = self.mesh_vertices[faces]

        # Get the valid spawn points on the terrain
        _, self.candidate_spawns = trim_boundary_aabb(self.mesh_vertices, 20.0)
        _, self.valid_spawns = sample_min_separation_xy(self.candidate_spawns, self.cfg.scene.num_envs, 30.0)
        self.valid_spawns[:,2] += 0.05

        # Plot Vertices of Terrain Mesh (Only Needed for Debugging)
        # vertex_cfg = MESH_MARKERS_CFG.copy()
        # vertex_cfg.prim_path = "/Visuals/spawns"
        # self.vertex_markers = VisualizationMarkers(cfg=vertex_cfg)
        # self.vertex_markers.set_visibility(True)
        # self.vertex_markers.visualize(self.valid_spawns2)

    def _get_goal_vec_normalized(self):
    
        goal_vec = self.target_spawns - self.robot.data.root_pos_w  
        return goal_vec/torch.linalg.norm(goal_vec, dim=-1, keepdim=True)

    def _update_marker_yaws(self):

        self.commands = self._get_goal_vec_normalized()

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
        self.goal_markers.visualize(self.target_spawns, marker_indices=indices)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._visualize_markers()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)
        self.scene.write_data_to_sim()

    def _get_observations(self) -> dict:

        self.scene.update(1.0/120.0)

        ''' Get and Process the Camera Reading '''
        camera_data = self.robot_camera.data.output["rgb"] / 255.0
        # normalize the camera data for better training results
        mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
        camera_data -= mean_tensor

        # On first call, fill history with the current frame
        if self._camera_hist is None:
            # repeat the first frame history_len times

            camera_data = camera_data.unsqueeze(1)
            camera_data = camera_data.repeat(1, self.history_len, 1, 1, 1)
            self._camera_hist = camera_data

        else:
            # drop oldest frame and append newest
            # _camera_hist[:, 1:] are t-3…t, so cat with new frame at dim=1
            new = camera_data.unsqueeze(1)   # (N,1,H,W,C)
            self._camera_hist = torch.cat([self._camera_hist[:, 1:], new], dim=1)

        N, T, H, W, C = self._camera_hist.shape

        ''' Get and Process the GPS Reading '''
        goal_vec = self.target_spawns - self.robot.data.root_pos_w
        goal_dist = torch.linalg.norm(goal_vec, dim=-1, keepdim=True)
        unit_goal = self._get_goal_vec_normalized()
        state_input = torch.hstack((unit_goal, goal_dist))
        _, S = state_input.shape
        self.state_input = state_input[:, None, None, None, :].expand(N, T, H, W, S)

        final_obs = torch.cat([self._camera_hist.clone(), self.state_input.clone()], dim=-1) # (N,T,H,W,C+4)

        return {"policy": final_obs}
    
    def _get_rewards(self) -> torch.Tensor:

        # Angle Alignment Reward
        goal_vec = self.target_spawns - self.robot.data.root_pos_w 
        goal_vec[:,2] = 0.0
        goal_vec = goal_vec/torch.linalg.norm(goal_vec, dim=-1, keepdim=True)
        forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)
        forwards[:,2] = 0.0
        forwards = forwards/torch.linalg.norm(forwards, dim=-1, keepdim=True)
        alignment = torch.sum(forwards * goal_vec, dim=-1, keepdim=True)
        angle = torch.acos(alignment.clamp(-1.0, 1.0))
        vel = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1) # torch.Size([N, 1])
        threshold = math.pi / 9.0
        mask = (angle <= threshold).to(torch.float32) # torch.Size([N, 1])
        forward_reward = vel * mask # torch.Size([N, 1])
        alignment_reward = forward_reward*torch.exp(alignment) # torch.Size([N, 1])

        # Arrival bonus, scaled by time remaining in episode
        vec3D   = self.target_spawns - self.robot.data.root_pos_w         # (N, 3)
        dist2D  = torch.linalg.norm(vec3D[:, :2], dim=-1, keepdim=True)   # (N, 1)
        remaining = (self.max_episode_length - self.episode_length_buf).clamp(min=0)
        remaining = remaining.to(self.gpu, dtype=torch.float32).unsqueeze(-1)
        frac_left = remaining / float(self.max_episode_length)
        remaining_time_gain = 1.0 + 1.0 * frac_left
        arrival_bonus = (dist2D < 0.5).float() * (2.0 * remaining_time_gain)                    # (N, 1)

        # Distance Penalty
        dist3D  = torch.linalg.norm(vec3D, dim=-1, keepdim=True)          # (N, 1)
        dist_penalty = -0.1 * dist3D

        # Penalize Flipping/Tipping Over
        gravity_vec = self.robot.data.projected_gravity_b
        flipped = (gravity_vec[:, 2] > 0.0).unsqueeze(-1)
        flip_penalty = (-5.0) * flipped.to(torch.float32)

        total_reward = alignment_reward + arrival_bonus + dist_penalty + flip_penalty                              # torch.Size([N, 1])

        return total_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        time_out = self.episode_length_buf >= (self.max_episode_length - 1)

        dist = self.target_spawns - self.robot.data.root_pos_w 
        dist_xy = torch.linalg.norm(dist[:, :2], dim=-1)
        reached = dist_xy < 0.5

        flipped = self.robot.data.projected_gravity_b[:, 2] > 0.0

        terminated = reached | flipped

        return terminated, time_out
    

    @torch.no_grad()    
    def _nearest_vertex_heights_from_xy(self, targets_xy: torch.Tensor,
                                        verts_xyz: torch.Tensor,
                                        chunk_size: int = 100_000) -> torch.Tensor:
        """
        For each (x,y) in targets_xy, find the closest vertex in verts_xyz (by L2 in XY)
        and return its z. All tensors must be on the same device.
        """
        device = targets_xy.device
        verts_xy = verts_xyz[:, :2]                               # (M, 2)
        N = targets_xy.shape[0]
        M = verts_xy.shape[0]

        best_d2  = torch.full((N,), float("inf"), device=device)
        best_idx = torch.full((N,), -1, dtype=torch.long, device=device)

        offset = 0
        while offset < M:
            end = min(offset + chunk_size, M)
            chunk = verts_xy[offset:end]                          # (C, 2)

            # (N, C) squared distances without sqrt for speed
            diff  = targets_xy[:, None, :] - chunk[None, :, :]    # (N, C, 2)
            d2    = (diff * diff).sum(dim=2)                      # (N, C)

            # best in this chunk
            min_d2, min_j = torch.min(d2, dim=1)                  # (N,), (N,)

            # keep global best
            better = min_d2 < best_d2
            best_d2[better]  = min_d2[better]
            best_idx[better] = min_j[better] + offset

            offset = end

        return verts_xyz[best_idx, 2]                              # (N,)
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] = self.valid_spawns[env_ids]
        default_root_state[:, 2] += 0.075
        self.robot.write_root_state_to_sim(default_root_state, env_ids)

        angles = torch.empty(len(env_ids), device=self.gpu).uniform_(-math.pi/8.0, math.pi/8.0)
        self.goal_radii[env_ids] = self.goal_radii[env_ids].uniform_(5.0, 6.0)

        targets = default_root_state[:, :3].clone()
        targets[:, 0] = targets[:, 0] + self.goal_radii[env_ids] * torch.cos(angles)
        targets[:, 1] = targets[:, 1] + self.goal_radii[env_ids] * torch.sin(angles)
        targets_xy = targets[:, :2]
        targets[:, 2] = self._nearest_vertex_heights_from_xy(targets_xy, self.mesh_vertices)
        targets[:, 2] += 0.5
        self.target_spawns[env_ids] = targets

        self._visualize_markers()
