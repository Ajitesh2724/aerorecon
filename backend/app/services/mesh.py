"""Surface mesh reconstruction from dense 3D point clouds."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger("aerorecon.mesh")


def generate_surface_mesh(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    output_obj_path: str,
    max_faces: int = 200_000,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """Generate a continuous 3D surface mesh from points using 2.5D Delaunay triangulation.

    Ideal for aerial drone mapping, terrain surveys, and infrastructure facades.
    Exports Wavefront OBJ format.
    """
    out_obj = Path(output_obj_path)
    out_obj.parent.mkdir(parents=True, exist_ok=True)

    n_pts = len(points)
    if n_pts < 3:
        raise ValueError(f"Need at least 3 points for surface mesh, got {n_pts}.")

    if on_progress:
        on_progress(20, f"Triangulating 3D surface from {n_pts:,} points…")

    # Subsample if extremely large to prevent OOM
    if n_pts > 100_000:
        step = int(np.ceil(n_pts / 100_000))
        pts = points[::step].astype(np.float32)
        cols = colors[::step].astype(np.uint8) if colors is not None else None
    else:
        pts = points.astype(np.float32)
        cols = colors.astype(np.uint8) if colors is not None else None

    # Perform 2D Delaunay triangulation in the dominant projection plane (XY for aerial)
    # Using scipy.spatial.Delaunay if available, or grid-based triangulation fallback
    faces = _triangulate_points(pts)

    if on_progress:
        on_progress(60, f"Filtering degenerate faces ({len(faces):,} triangles)…")

    # Filter out elongated / edge triangles exceeding maximum edge length
    faces = _filter_long_edges(pts, faces, max_edge_ratio=0.15)

    if len(faces) > max_faces:
        faces = faces[:max_faces]

    if on_progress:
        on_progress(85, "Writing Wavefront OBJ mesh…")

    _write_obj(out_obj, pts, cols, faces)

    if on_progress:
        on_progress(100, "Mesh generation complete.")

    stats = {
        "vertex_count": len(pts),
        "face_count": len(faces),
        "obj_size_mb": round(out_obj.stat().st_size / (1024 * 1024), 2),
    }

    logger.info("Mesh generated with %d vertices and %d faces (%s)", len(pts), len(faces), out_obj)
    return {
        "success": True,
        "mesh_obj": str(out_obj),
        "stats": stats,
    }


def _triangulate_points(pts: np.ndarray) -> np.ndarray:
    """Triangulate points projected on XY plane."""
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(pts[:, :2])
        return tri.simplices.astype(np.int32)
    except Exception as exc:
        logger.info("SciPy Delaunay not available (%s); using grid-based triangulation.", exc)
        return _grid_triangulation_fallback(pts)


def _grid_triangulation_fallback(pts: np.ndarray, grid_res: int = 150) -> np.ndarray:
    """Fast grid-based 2.5D surface triangulation fallback."""
    x = pts[:, 0]
    y = pts[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    gx = np.clip(((x - xmin) / (xmax - xmin + 1e-6) * (grid_res - 1)).astype(int), 0, grid_res - 1)
    gy = np.clip(((y - ymin) / (ymax - ymin + 1e-6) * (grid_res - 1)).astype(int), 0, grid_res - 1)

    grid = {}
    for idx, (ix, iy) in enumerate(zip(gx, gy)):
        grid[(ix, iy)] = idx

    faces = []
    for ix in range(grid_res - 1):
        for iy in range(grid_res - 1):
            p00 = grid.get((ix, iy))
            p10 = grid.get((ix + 1, iy))
            p01 = grid.get((ix, iy + 1))
            p11 = grid.get((ix + 1, iy + 1))

            if p00 is not None and p10 is not None and p11 is not None:
                faces.append([p00, p10, p11])
            if p00 is not None and p11 is not None and p01 is not None:
                faces.append([p00, p11, p01])

    if not faces:
        # Trivial triangle
        return np.array([[0, 1, 2]], dtype=np.int32)

    return np.array(faces, dtype=np.int32)


def _filter_long_edges(pts: np.ndarray, faces: np.ndarray, max_edge_ratio: float = 0.2) -> np.ndarray:
    """Filter out triangles with edges exceeding a threshold to remove spurious border connections."""
    if len(faces) == 0:
        return faces

    bbox_diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    max_len = max(0.5, bbox_diag * max_edge_ratio)

    v0 = pts[faces[:, 0]]
    v1 = pts[faces[:, 1]]
    v2 = pts[faces[:, 2]]

    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)

    valid = (e0 < max_len) & (e1 < max_len) & (e2 < max_len)
    filtered = faces[valid]
    return filtered if len(filtered) > 0 else faces


def _write_obj(filepath: Path, pts: np.ndarray, cols: Optional[np.ndarray], faces: np.ndarray) -> None:
    """Write Wavefront OBJ file with vertex coordinates, colors, and face indices."""
    has_cols = cols is not None and len(cols) == len(pts)

    with open(filepath, "w") as f:
        f.write("# AeroRecon 3D Surface Mesh\n")
        f.write(f"# Vertices: {len(pts)}, Faces: {len(faces)}\n")

        # Write vertices (OBJ indices are 1-based)
        for i in range(len(pts)):
            p = pts[i]
            if has_cols:
                c = cols[i] / 255.0  # OBJ vertex colors in [0, 1]
                f.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
            else:
                f.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

        # Write faces
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
