"""Utility and processing package for Image-Resizer.

This package contains:
- io_utils: filesystem and image I/O helpers
- processing: image processing algorithms
- metrics: helpers to persist params and run metrics with history
"""

from .io_utils import (
    ensure_dir,
    is_image_file,
    discover_images,
    make_output_path,
    load_image_cv2,
    save_image,
)

from .processing import (
    anisotropic_diffusion_gray,
    anisotropic_diffusion_color,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    reconstruct_from_laplacian,
    bspline_resize_channel,
    bspline_resize,
    compute_local_energy,
    bspline_resize_adaptive,
    resize_modern,
    process_one_image,
)

from .metrics import (
    save_params_json,
    append_run_metrics,
)

__all__ = [
    # io_utils
    "ensure_dir",
    "is_image_file",
    "discover_images",
    "make_output_path",
    "load_image_cv2",
    "save_image",
    # processing
    "anisotropic_diffusion_gray",
    "anisotropic_diffusion_color",
    "build_gaussian_pyramid",
    "build_laplacian_pyramid",
    "reconstruct_from_laplacian",
    "bspline_resize_channel",
    "bspline_resize",
    "compute_local_energy",
    "bspline_resize_adaptive",
    "resize_modern",
    "process_one_image",
    # metrics
    "save_params_json",
    "append_run_metrics",
]
