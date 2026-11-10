#!/usr/bin/env python3
"""Генератор процедурных текстур для лабораторной работы №3.

Создаёт в каталоге textures/ наборы <имя>_albedo.png, <имя>_specular.png и
(для светящихся материалов) <имя>_emissive.png. Шум Перлина периодический, поэтому
текстуры бесшовно тайлятся (режим адресации REPEAT).

Запуск: python tools/generate_textures.py   (нужны numpy и pillow)
"""
import os
import numpy as np
from PIL import Image

SIZE = 512
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "textures")


def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin(size, period, rng):
    """Периодический шум Перлина, значения примерно в [-1, 1]."""
    ang = rng.uniform(0, 2 * np.pi, (period, period))
    gx, gy = np.cos(ang), np.sin(ang)
    lin = np.arange(size) / size * period
    x, y = np.meshgrid(lin, lin)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0

    def dot(ix, iy, dx, dy):
        return gx[iy % period, ix % period] * dx + gy[iy % period, ix % period] * dy

    n00 = dot(x0, y0, fx, fy)
    n10 = dot(x0 + 1, y0, fx - 1, fy)
    n01 = dot(x0, y0 + 1, fx, fy - 1)
    n11 = dot(x0 + 1, y0 + 1, fx - 1, fy - 1)
    u, v = fade(fx), fade(fy)
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    return (nx0 + v * (nx1 - nx0)) * 1.41


def fbm(size, base_period, octaves, rng):
    total, amp, norm = np.zeros((size, size)), 1.0, 0.0
    for o in range(octaves):
        total += amp * perlin(size, base_period * 2 ** o, rng)
        norm += amp
        amp *= 0.5
    return total / norm


def save(name, kind, arr):
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    img = Image.fromarray((arr * 255).astype(np.uint8), "RGB")
    img.save(os.path.join(OUT, f"{name}_{kind}.png"))


def bricks(rng):
    rows, bricks_per_row, mortar = 8, 4, 7
    bh, bw = SIZE // rows, SIZE // bricks_per_row
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    row = yy // bh
    shift = (row % 2) * (bw // 2)
    xs = (xx + shift) % SIZE
    col = xs // bw
    in_x, in_y = xs % bw, yy % bh
    mortar_mask = (in_x < mortar) | (in_y < mortar)
    brick_id = (row * 7 + col * 13) % 11
    tint = 0.75 + 0.25 * (brick_id / 10.0)
    grain = fbm(SIZE, 8, 4, rng)
    base = np.array([0.62, 0.25, 0.18])
    col_img = base[None, None, :] * tint[..., None] * (0.85 + 0.35 * grain[..., None])
    mortar_col = np.array([0.62, 0.60, 0.56]) * (0.85 + 0.2 * grain[..., None])
    albedo = np.where(mortar_mask[..., None], mortar_col, col_img)
    spec = np.where(mortar_mask, 0.0, 0.25 + 0.15 * (grain * 0.5 + 0.5))
    save("bricks", "albedo", albedo)
    save("bricks", "specular", spec)


def stone(rng):
    n, grout = 4, 6
    t = SIZE // n
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    gx, gy = xx % t, yy % t
    grout_mask = (gx < grout) | (gy < grout)
    tile_id = ((xx // t) * 5 + (yy // t) * 3) % 7
    tint = 0.7 + 0.3 * tile_id / 6.0
    veins = fbm(SIZE, 4, 5, rng)
    vein = np.abs(veins) < 0.04
    base = np.array([0.50, 0.52, 0.56])
    col = base[None, None, :] * tint[..., None] * (0.9 + 0.3 * veins[..., None])
    col = np.where(vein[..., None], col * 1.5, col)
    grout_col = np.array([0.18, 0.18, 0.2])
    albedo = np.where(grout_mask[..., None], grout_col, col)
    spec = np.where(grout_mask, 0.0, 0.85 - 0.3 * (veins * 0.5 + 0.5))
    save("stone", "albedo", albedo)
    save("stone", "specular", spec)


def crate(rng):
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    frame = 48
    border = (xx < frame) | (xx >= SIZE - frame) | (yy < frame) | (yy >= SIZE - frame)
    plank_h = (SIZE - 2 * frame) // 4
    plank = np.clip((yy - frame) // plank_h, 0, 3)
    seam = ((yy - frame) % plank_h < 3) & ~border
    grain = fbm(SIZE, 3, 3, rng)
    stripes = 0.5 + 0.5 * np.sin((yy * 0.35 + grain * 40.0) + plank * 2.0)
    light = np.array([0.72, 0.50, 0.26])
    dark = np.array([0.42, 0.27, 0.12])
    wood = dark[None, None, :] + (light - dark)[None, None, :] * stripes[..., None]
    frame_col = wood * 0.75
    img = np.where(border[..., None], frame_col, wood)
    img = np.where(seam[..., None], img * 0.35, img)
    spec = np.where(seam, 0.0, 0.18 + 0.12 * stripes)
    save("crate", "albedo", img)
    save("crate", "specular", spec)


def magma(rng):
    rock = fbm(SIZE, 4, 5, rng)
    ridged = 1.0 - np.abs(fbm(SIZE, 3, 4, rng)) * 2.2        # светлые "трещины"
    cracks = np.clip((ridged - 0.62) / 0.3, 0.0, 1.0) ** 1.5   # маска трещин
    rock_col = np.array([0.12, 0.10, 0.10])[None, None, :] * (0.7 + 0.8 * (rock[..., None] * 0.5 + 0.5))
    lava = np.array([1.0, 0.38, 0.05])[None, None, :]
    albedo = rock_col * (1 - cracks[..., None]) + lava * cracks[..., None] * 0.9
    emissive = lava * (cracks[..., None] ** 1.2) * (0.7 + 0.5 * (rng.uniform(size=(1, 1, 1)) * 0 + 1))
    spec = 0.5 * (1 - cracks)
    save("magma", "albedo", albedo)
    save("magma", "specular", spec)
    save("magma", "emissive", emissive)


def checker(rng):
    n = 8
    t = SIZE // n
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    c = ((xx // t + yy // t) % 2).astype(float)
    albedo = np.stack([0.15 + 0.75 * c, 0.2 + 0.7 * c, 0.9 - 0.7 * c], axis=-1)
    save("checker", "albedo", albedo)   # без specular/emissive -> служебные чёрные текстуры


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(12345)
    for fn in (bricks, stone, crate, magma, checker):
        fn(rng)
    print("Текстуры созданы в", os.path.abspath(OUT))
