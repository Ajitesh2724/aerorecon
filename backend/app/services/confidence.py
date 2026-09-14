"""Multi-factor confidence mapping and uncertainty estimation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger("aerorecon.confidence")


def compute_confidence_map(
    points_count: int,
    registered_images: int,
    mean_reprojection_error: Optional[float] = None,
    mean_flow_consistency: Optional[float] = None,
    has_gps: bool = False,
    output_dir: Optional[str] = None,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """Calculate multi-tier confidence metrics for the 3D reconstruction.

    Confidence scoring factors:
      1. View redundancy & multi-view baseline
      2. Reprojection error residuals
      3. Optical flow forward-backward consistency
      4. Telemetry and georeferencing availability
    """
    if on_progress:
        on_progress(20, "Evaluating multi-view geometric confidence…")

    # 1. Geometry redundancy score [0..1]
    # Optimal drone coverage: at least 15-30 registered frames
    view_score = min(1.0, registered_images / 25.0)

    # 2. Precision score from reprojection error [0..1]
    if mean_reprojection_error is not None:
        # Error <= 0.8px is excellent (1.0), >= 3.0px is poor (0.0)
        reproj_score = float(np.clip(1.0 - (mean_reprojection_error - 0.5) / 2.5, 0.0, 1.0))
    else:
        reproj_score = 0.75  # default estimate

    # 3. Motion consistency score [0..1]
    flow_score = float(mean_flow_consistency) if mean_flow_consistency is not None else 0.85

    # 4. Geodetic prior score [0..1]
    gps_score = 1.0 if has_gps else 0.5

    if on_progress:
        on_progress(60, "Computing confidence tier distribution…")

    # Composite confidence score
    composite = (
        0.35 * view_score +
        0.30 * reproj_score +
        0.20 * flow_score +
        0.15 * gps_score
    )

    # Calculate 4-tier distribution percentages
    high_pct = float(np.clip(composite * 85.0 + 5.0, 10.0, 92.0))
    rem = 100.0 - high_pct
    med_pct = rem * 0.70
    low_pct = rem * 0.22
    unseen_pct = rem * 0.08

    distribution = {
        "high": round(high_pct, 1),
        "medium": round(med_pct, 1),
        "low": round(low_pct, 1),
        "unseen": round(unseen_pct, 1),
    }

    overall_tier = "high" if composite >= 0.75 else "medium" if composite >= 0.50 else "low"

    factors = {
        "view_redundancy": round(view_score, 2),
        "reprojection_precision": round(reproj_score, 2),
        "flow_consistency": round(flow_score, 2),
        "georeferencing_availability": round(gps_score, 2),
        "composite_score": round(composite, 3),
    }

    report = {
        "overall_tier": overall_tier,
        "composite_score": round(composite, 3),
        "distribution": distribution,
        "factors": factors,
        "is_estimated": True,
        "points_evaluated": points_count,
        "views_evaluated": registered_images,
    }

    out_file = None
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        out_file = out_p / "confidence_report.json"
        out_file.write_text(json.dumps(report, indent=2))

    if on_progress:
        on_progress(100, "Confidence mapping complete.")

    logger.info("Confidence assessment: %s (composite score: %.2f)", overall_tier.upper(), composite)
    return {
        "success": True,
        "report_path": str(out_file) if out_file else None,
        "summary": report,
    }
