from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline

from .io_utils import load_image_cv2


def anisotropic_diffusion_gray(
    image: np.ndarray,
    niter: int = 10,
    kappa: float = 30.0,
    gamma: float = 0.15,
    option: int = 1,
) -> np.ndarray:
    img = image.astype(np.float32)
    for _ in range(niter):
        north = np.zeros_like(img)
        south = np.zeros_like(img)
        east = np.zeros_like(img)
        west = np.zeros_like(img)

        north[1:, :] = img[1:, :] - img[:-1, :]
        south[:-1, :] = img[:-1, :] - img[1:, :]
        east[:, :-1] = img[:, :-1] - img[:, 1:]
        west[:, 1:] = img[:, 1:] - img[:, :-1]

        if option == 1:
            cN = np.exp(-(north / kappa) ** 2.0)
            cS = np.exp(-(south / kappa) ** 2.0)
            cE = np.exp(-(east / kappa) ** 2.0)
            cW = np.exp(-(west / kappa) ** 2.0)
        else:
            cN = 1.0 / (1.0 + (north / kappa) ** 2.0)
            cS = 1.0 / (1.0 + (south / kappa) ** 2.0)
            cE = 1.0 / (1.0 + (east / kappa) ** 2.0)
            cW = 1.0 / (1.0 + (west / kappa) ** 2.0)

        img += gamma * (
            cN * north + cS * south + cE * east + cW * west
        )
    return img


def anisotropic_diffusion_color(
    image_rgb: np.ndarray,
    niter: int = 10,
    kappa: float = 30.0,
    gamma: float = 0.15,
    option: int = 1,
) -> np.ndarray:
    r, g, b = cv2.split(image_rgb.astype(np.float32))
    r = anisotropic_diffusion_gray(r, niter=niter, kappa=kappa, gamma=gamma, option=option)
    g = anisotropic_diffusion_gray(g, niter=niter, kappa=kappa, gamma=gamma, option=option)
    b = anisotropic_diffusion_gray(b, niter=niter, kappa=kappa, gamma=gamma, option=option)
    return cv2.merge((r, g, b))


def build_gaussian_pyramid(image: np.ndarray, levels: int = 4):
    g = [image]
    for _ in range(1, levels):
        image = cv2.pyrDown(image)
        g.append(image)
    return g


def build_laplacian_pyramid(gaussian_pyr: list):
    l = [gaussian_pyr[-1]]
    for i in range(len(gaussian_pyr) - 1, 0, -1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        ge = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
        l.append(cv2.subtract(gaussian_pyr[i - 1], ge))
    return l[::-1]


def reconstruct_from_laplacian(laplacian_pyr: list):
    current = laplacian_pyr[-1]
    for i in range(len(laplacian_pyr) - 2, -1, -1):
        size = (laplacian_pyr[i].shape[1], laplacian_pyr[i].shape[0])
        current = cv2.pyrUp(current, dstsize=size)
        current = cv2.add(current, laplacian_pyr[i])
    return current


def bspline_resize_channel(channel: np.ndarray, scale_x: float, scale_y: float, order: int = 3) -> np.ndarray:
    h, w = channel.shape
    new_h = max(1, int(round(h * scale_y)))
    new_w = max(1, int(round(w * scale_x)))

    y = np.arange(h)
    x = np.arange(w)
    spline = RectBivariateSpline(y, x, channel.astype(np.float64), kx=order, ky=order)

    y_new = np.linspace(0, h - 1, new_h)
    x_new = np.linspace(0, w - 1, new_w)
    out = spline(y_new, x_new)
    return out.astype(channel.dtype)


def bspline_resize(image: np.ndarray, scale_x: float, scale_y: float, order: int = 3) -> np.ndarray:
    if image.ndim == 2:
        return bspline_resize_channel(image, scale_x, scale_y, order)
    c = []
    for i in range(image.shape[2]):
        c.append(bspline_resize_channel(image[:, :, i], scale_x, scale_y, order))
    return np.stack(c, axis=2)


def compute_local_energy(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = cv2.GaussianBlur(mag, (3, 3), 0)
    return mag


def bspline_resize_adaptive(
    image: np.ndarray,
    scale_x: float,
    scale_y: float,
    order_low: int = 1,
    order_high: int = 3,
    beta: float = 2.0,
) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        gray = image.astype(np.float32) / 255.0
    energy = compute_local_energy(gray)
    energy_resized = cv2.resize(energy, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    e_norm = energy_resized / (energy_resized.max() + 1e-6)
    w = np.clip(e_norm ** beta, 0.0, 1.0)

    low = bspline_resize(image, scale_x, scale_y, order=order_low).astype(np.float32)
    high = bspline_resize(image, scale_x, scale_y, order=order_high).astype(np.float32)
    out = (1.0 - w[..., None]) * low + w[..., None] * high if image.ndim == 3 else (1.0 - w) * low + w * high
    return out.astype(image.dtype)


def resize_modern(
    image: np.ndarray,
    scale_x: float,
    scale_y: float,
    spline_order: int = 3,
    diffusion_iter: int = 10,
    pyr_levels: int = 4,
    kappa: float = 30.0,
    gamma: float = 0.15,
    option: int = 1,
    detail_boost: float = 1.0,
    sharpen: bool = True,
    compression_factor: float = 1.0,
) -> np.ndarray:
    if compression_factor is not None and compression_factor > 1.0:
        scale_x = 1.0
        scale_y = 1.0

    img = image

    if diffusion_iter and diffusion_iter > 0:
        img = anisotropic_diffusion_color(img, diffusion_iter, kappa, gamma, option)
        img = np.clip(img, 0.0, 1.0)

    if detail_boost != 1.0 or sharpen:
        g = build_gaussian_pyramid(img, levels=max(2, pyr_levels))
        l = build_laplacian_pyramid(g)
        if detail_boost != 1.0:
            for i in range(0, len(l) - 1):
                l[i] = l[i] * float(detail_boost)
        img = reconstruct_from_laplacian(l)
        img = np.clip(img, 0.0, 1.0)

    out = bspline_resize(img, scale_x, scale_y, order=spline_order)

    if sharpen:
        k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(out, -1, k)
        out = np.clip(out, 0.0, 1.0)
    return out


def process_one_image(
    input_image_path: Path,
    scale_x: float,
    scale_y: float,
    spline_order: int = 3,
    diffusion_iter: int = 10,
    pyr_levels: int = 4,
    kappa: float = 30.0,
    gamma: float = 0.15,
    option: int = 1,
    detail_boost: float = 1.0,
    sharpen: bool = True,
    compression_factor: float = 1.0,
) -> np.ndarray:
    img = load_image_cv2(str(input_image_path))
    out = resize_modern(
        img,
        scale_x=scale_x,
        scale_y=scale_y,
        spline_order=spline_order,
        diffusion_iter=diffusion_iter,
        pyr_levels=pyr_levels,
        kappa=kappa,
        gamma=gamma,
        option=option,
        detail_boost=detail_boost,
        sharpen=sharpen,
        compression_factor=compression_factor,
    )
    return out
