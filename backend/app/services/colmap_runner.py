"""COLMAP runner alias — imports functionality from sfm.py."""

from .sfm import (
    get_colmap_path,
    run_colmap_sfm,
    run_opencv_sfm,
)

__all__ = ["get_colmap_path", "run_colmap_sfm", "run_opencv_sfm"]
