"""
Compositional 3DGS scene: environment + object Gaussian clusters.

Manages a static environment (empty table, floor, robot) and a set of
movable object Gaussian clusters. At render time, concatenates everything
into a single set of Gaussians for gsplat.

Usage:
    scene = CompositionalScene.from_ply(env_ply_path)
    scene.add_object("mug", mug_cluster, position=[0.1, 0.05, 0.03])
    scene.add_object("can", can_cluster, position=[-0.1, 0.0, 0.05])

    # Move an object (e.g., during trajectory)
    scene.set_object_pose("mug", new_position, new_rotation_quat)

    # Get concatenated Gaussians for rendering
    means, colors, opacities, scales, quats = scene.get_render_tensors(device="cuda")
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional

from src.pipeline.gaussian_transform import (
    GaussianCluster, transform_gaussians, quat_to_rotation_matrix,
)


def load_gaussian_cluster_from_ply(
    ply_path: str,
    classifier_path: Optional[str] = None,
    object_id: Optional[int] = None,
    threshold: float = 0.3,
    max_points: int = 100000,
) -> GaussianCluster:
    """Load a Gaussian cluster from a PLY file.

    If classifier_path and object_id are given, extracts only the Gaussians
    belonging to that object. Otherwise loads all Gaussians.
    """
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    n = len(v.data)

    # Extract all Gaussian properties
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    quaternions = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32)
    scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32)
    opacities = np.array(v["opacity"]).astype(np.float32)
    sh_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)

    # Filter by object ID if requested
    if classifier_path is not None and object_id is not None:
        import torch as th
        features = np.stack([v[f"obj_dc_{i}"] for i in range(16)], axis=-1)
        classifier = th.load(classifier_path, map_location="cpu", weights_only=True)
        weight = classifier["weight"].squeeze(-1).squeeze(-1)
        bias = classifier["bias"]

        with th.no_grad():
            feat_tensor = th.tensor(features, dtype=th.float32)
            logits = feat_tensor @ weight.T + bias
            probs = th.softmax(logits, dim=-1)
            obj_prob = probs[:, object_id].numpy()

        mask = obj_prob > threshold
        positions = positions[mask]
        quaternions = quaternions[mask]
        scales = scales[mask]
        opacities = opacities[mask]
        sh_dc = sh_dc[mask]

    # Subsample if too many
    if len(positions) > max_points:
        idx = np.random.choice(len(positions), max_points, replace=False)
        positions = positions[idx]
        quaternions = quaternions[idx]
        scales = scales[idx]
        opacities = opacities[idx]
        sh_dc = sh_dc[idx]

    return GaussianCluster(
        positions=positions,
        quaternions=quaternions,
        scales=scales,
        opacities=opacities,
        sh_dc=sh_dc,
    )


class CompositionalScene:
    """A scene composed of a static environment + movable object clusters."""

    def __init__(self, environment: GaussianCluster):
        """Initialize with a static environment (table, floor, etc.).

        Args:
            environment: GaussianCluster for the static background.
        """
        self.environment = environment
        self.objects: dict[str, GaussianCluster] = {}         # original clusters
        self.object_anchors: dict[str, np.ndarray] = {}       # original centroid
        self.object_current_pos: dict[str, np.ndarray] = {}   # current position
        self.object_current_quat: dict[str, np.ndarray] = {}  # current rotation delta

    @classmethod
    def from_ply(
        cls,
        ply_path: str,
        classifier_path: Optional[str] = None,
        exclude_object_ids: Optional[list[int]] = None,
        env_threshold: float = 0.3,
    ) -> "CompositionalScene":
        """Load environment from PLY, optionally excluding certain object IDs.

        Args:
            ply_path: Path to the full scene PLY.
            classifier_path: Path to classifier for object filtering.
            exclude_object_ids: Object IDs to exclude from the environment
                (these will be loaded as separate objects).
            env_threshold: Classification threshold.
        """
        # Load all Gaussians (no subsampling) so mask indices align with classifier
        env_cluster = load_gaussian_cluster_from_ply(ply_path, max_points=999999999)

        if exclude_object_ids and classifier_path:
            # Remove specified objects from environment
            import torch as th
            from plyfile import PlyData

            ply = PlyData.read(ply_path)
            v = ply["vertex"]
            features = np.stack([v[f"obj_dc_{i}"] for i in range(16)], axis=-1)
            classifier = th.load(classifier_path, map_location="cpu", weights_only=True)
            weight = classifier["weight"].squeeze(-1).squeeze(-1)
            bias = classifier["bias"]

            with th.no_grad():
                feat_tensor = th.tensor(features, dtype=th.float32)
                logits = feat_tensor @ weight.T + bias
                pred_ids = logits.argmax(dim=-1).numpy()

            # Keep only Gaussians NOT belonging to excluded objects
            keep_mask = np.ones(len(pred_ids), dtype=bool)
            for oid in exclude_object_ids:
                keep_mask &= (pred_ids != oid)

            env_cluster = GaussianCluster(
                positions=env_cluster.positions[keep_mask],
                quaternions=env_cluster.quaternions[keep_mask],
                scales=env_cluster.scales[keep_mask],
                opacities=env_cluster.opacities[keep_mask],
                sh_dc=env_cluster.sh_dc[keep_mask],
            )

        return cls(env_cluster)

    def add_object(
        self,
        name: str,
        cluster: GaussianCluster,
        position: Optional[np.ndarray] = None,
    ):
        """Add a movable object to the scene.

        Args:
            name: Unique identifier for this object.
            cluster: The object's Gaussian cluster.
            position: Initial world position. If None, uses the cluster's centroid.
        """
        self.objects[name] = cluster
        centroid = cluster.centroid
        self.object_anchors[name] = centroid.copy()

        if position is not None:
            self.object_current_pos[name] = np.asarray(position, dtype=np.float32)
        else:
            self.object_current_pos[name] = centroid.copy()

        self.object_current_quat[name] = np.array([1, 0, 0, 0], dtype=np.float32)

    def set_object_pose(
        self,
        name: str,
        position: np.ndarray,
        rotation_quat: Optional[np.ndarray] = None,
    ):
        """Update an object's world pose.

        Args:
            name: Object identifier.
            position: New world position for the object centroid.
            rotation_quat: Rotation delta quaternion (wxyz). None = no rotation.
        """
        self.object_current_pos[name] = np.asarray(position, dtype=np.float32)
        if rotation_quat is not None:
            self.object_current_quat[name] = np.asarray(rotation_quat, dtype=np.float32)

    def get_transformed_object(self, name: str) -> GaussianCluster:
        """Get a copy of the object's Gaussians at their current pose."""
        original = self.objects[name]
        anchor = self.object_anchors[name]
        new_pos = self.object_current_pos[name]
        rot_quat = self.object_current_quat[name]

        return transform_gaussians(original, anchor, new_pos, rot_quat)

    def get_all_gaussians(self) -> GaussianCluster:
        """Concatenate environment + all objects into one cluster for rendering."""
        clusters = [self.environment]
        for name in self.objects:
            clusters.append(self.get_transformed_object(name))

        return GaussianCluster(
            positions=np.concatenate([c.positions for c in clusters]),
            quaternions=np.concatenate([c.quaternions for c in clusters]),
            scales=np.concatenate([c.scales for c in clusters]),
            opacities=np.concatenate([c.opacities for c in clusters]),
            sh_dc=np.concatenate([c.sh_dc for c in clusters]),
        )

    def get_render_tensors(self, device: str = "cuda"):
        """Get concatenated Gaussians as torch tensors ready for gsplat.

        Returns:
            means: (N, 3) positions
            colors: (N, 1, 3) SH DC coefficients
            opacities: (N,) sigmoid-space opacities
            scales: (N, 3) exp-space scales
            quats: (N, 4) normalized quaternions
        """
        all_g = self.get_all_gaussians()

        means = torch.tensor(all_g.positions, dtype=torch.float32, device=device)
        colors = torch.tensor(all_g.sh_dc, dtype=torch.float32, device=device).unsqueeze(1)
        opacities = torch.sigmoid(torch.tensor(all_g.opacities, dtype=torch.float32, device=device))
        scales = torch.exp(torch.tensor(all_g.scales, dtype=torch.float32, device=device))
        quats = torch.tensor(all_g.quaternions, dtype=torch.float32, device=device)
        quats = quats / quats.norm(dim=-1, keepdim=True)

        return means, colors, opacities, scales, quats

    def summary(self) -> str:
        lines = [f"CompositionalScene: {self.environment.n:,} env Gaussians"]
        for name, cluster in self.objects.items():
            pos = self.object_current_pos[name]
            lines.append(f"  {name}: {cluster.n:,} Gaussians at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        total = self.environment.n + sum(c.n for c in self.objects.values())
        lines.append(f"  Total: {total:,} Gaussians")
        return "\n".join(lines)
