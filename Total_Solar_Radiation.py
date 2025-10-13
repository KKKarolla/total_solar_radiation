import pygame
import random
import math
import time
try:
    import pandas as pd
except Exception:
    pd = None
import os
import csv

def draw_contour_lines(screen, width, height, center_x, center_y, layers=18, points_a=None, points_b=None, blend=1.0):
    """Draw contour rings that follow the point-cloud envelope and simulate rotation around Z.

    points: list of {'x','y','z'} or None. If provided, we compute an angular envelope so
    contours wrap the point-cluster like topographic rings. If None, falls back to concentric rings.
    """
    rot = time.time() * 0.6
    base_surf = pygame.Surface((width, height), pygame.SRCALPHA)

    # sampling resolution around circle
    waves = 480

    def compute_envelope(points):
        env = [0.0] * waves
        env_z = [0.0] * waves
        dens = [0.0] * waves
        if not points:
            for i in range(waves):
                env[i] = 80.0
            return env, env_z, dens

        bins_r = [0.0] * waves
        bins_z = [0.0] * waves
        bins_n = [0] * waves
        for p in points:
            dx = p['x'] - center_x
            dy = p['y'] - center_y
            r = math.hypot(dx, dy)
            theta = math.atan2(dy, dx)
            if theta < 0:
                theta += 2 * math.pi
            idx = int((theta / (2 * math.pi)) * waves) % waves
            bins_r[idx] += r
            bins_z[idx] += p.get('z', 0)
            bins_n[idx] += 1

        for i in range(waves):
            if bins_n[i] > 0:
                env[i] = bins_r[i] / bins_n[i]
                env_z[i] = bins_z[i] / bins_n[i]
                dens[i] = bins_n[i]
            else:
                env[i] = 80.0
                env_z[i] = 0.0
                dens[i] = 0.0

        # normalize density
        maxd = max(dens) or 1
        for i in range(waves):
            dens[i] = dens[i] / maxd

        # smooth arrays
        def smooth(arr, radius=3):
            out = [0.0] * waves
            for i in range(waves):
                s = 0.0; c = 0
                for j in range(i-radius, i+radius+1):
                    idx = j % waves
                    s += arr[idx]; c += 1
                out[i] = s / c
            return out

        env = smooth(env, radius=3)
        env_z = smooth(env_z, radius=3)
        return env, env_z, dens

    env_a, envz_a, dens_a = compute_envelope(points_a)
    env_b, envz_b, dens_b = compute_envelope(points_b)

    # blend envelopes by blend ratio (0..1)
    envelope = [ (1.0-blend)*env_a[i] + blend*env_b[i] for i in range(waves) ]
    envelope_z = [ (1.0-blend)*envz_a[i] + blend*envz_b[i] for i in range(waves) ]
    density = [ (1.0-blend)*dens_a[i] + blend*dens_b[i] for i in range(waves) ]

    # draw layers: inner layers drawn first
    for layer in range(layers):
        base_alpha = 30 + int((layers - layer) * (160 / layers))
        alpha = max(10, min(255, base_alpha))
        # use same pink as the points for waves (preserve alpha)
        color = (255, 80, 160, alpha)

        pts = []
        layer_scale = 1.0 + layer * 0.03
        layer_gap = 6 + layer * 1.2
        wiggle_amp = 4 + layer * 0.8

        for i in range(waves):
            theta = (i / waves) * (2 * math.pi)
            theta_rot = theta + rot * (0.08 + layer * 0.008)

            # use blended envelope (envelope/envelope_z/density computed above)
            env_r = envelope[i]
            env_z = envelope_z[i]
            dens = density[i]
            # bulge according to density and z: denser/higher => ring sits closer and higher
            bulge = (dens - 0.45) * 40.0 + (env_z * 0.08)
            r = env_r * layer_scale + layer * layer_gap + math.sin(theta * (1.6 + layer*0.05) + rot*(0.4+layer*0.02)) * (wiggle_amp)
            r += bulge
            # project vertical offset from env_z and layer to y for perspective
            y_off = -env_z * 0.12 - layer * 1.6

            x = int(center_x + r * math.cos(theta_rot))
            y = int(center_y + r * math.sin(theta_rot) + y_off)
            pts.append((x, y))

        try:
            pygame.draw.aalines(base_surf, color, True, pts)
        except Exception:
            pygame.draw.lines(base_surf, color, True, pts)

    # draw subtle multiple composites for softness
    screen.blit(base_surf, (0, 0))
    # light second pass with lower alpha to smooth appearance
    try:
        overlay = base_surf.copy()
        overlay.fill((255,255,255,0), None, pygame.BLEND_RGBA_MULT)
        screen.blit(overlay, (0,0))
    except Exception:
        pass

def lerp(a, b, t):
    return a + (b - a) * t

def generate_point_cloud(center_x, center_y, num_points=480, std_x=80, std_y=60, multiplier=1.0):
    points = []
    for i in range(num_points):
        x = int(random.gauss(center_x, std_x))
        y = int(random.gauss(center_y, std_y))
        dist = math.hypot(x - center_x, y - center_y)
        z = max(0, int(160 - dist + random.gauss(0, 24)) * multiplier)
        points.append({'x': x, 'y': y, 'z': z, 'phase': random.random()*math.pi*2})
    return points

def draw_point_cloud(screen, points_a, points_b, t, ratio):
    for i in range(len(points_a)):
        pa = points_a[i]
        pb = points_b[i]
        x = int(lerp(pa['x'], pb['x'], ratio))
        y = int(lerp(pa['y'], pb['y'], ratio))
        z = int(lerp(pa['z'], pb['z'], ratio))
        phase = lerp(pa['phase'], pb['phase'], ratio)
        dz = int(math.sin(t*3 + phase + x * 0.02) * 32)
        top = (x, y - (z + dz))
        # draw vertical connector in a pale cyan/blue; keep pink dots unchanged
        pygame.draw.line(screen, (180, 230, 255), (x, y), top, 2)
        radius = max(3, min(8, int(5 + dz/20)))
        pygame.draw.circle(screen, (255, 80, 160), top, radius)

def main_visualization():
    pygame.init()
    width, height = 1200, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Big Data Point Cloud Visualization')
    clock = pygame.time.Clock()
    center_x, center_y = width // 2, height // 2 + 80

    # 读取数据 (支持 pandas 或 csv 回退)
    csv_path = os.path.join(os.path.dirname(__file__), 'daily_KP_GSR_ALL.csv')
    yearly = {}
    years_sorted = []
    if os.path.exists(csv_path):
        if pd is not None:
            try:
                df = pd.read_csv(csv_path, sep=None, engine='python', on_bad_lines='skip')
                cols = list(df.columns)
                year_col = None
                val_col = None
                for c in cols:
                    lc = str(c).lower()
                    if 'year' in lc or '年' in lc:
                        year_col = c
                    if 'value' in lc or '數值' in lc or 'value' in lc:
                        val_col = c
                if year_col is not None and val_col is not None:
                    df = df.dropna(subset=[year_col, val_col])
                    df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
                    df = df.dropna(subset=[val_col])
                    grouped = df.groupby(year_col)[val_col].sum()
                    for y, v in grouped.items():
                        try:
                            yi = int(y)
                        except:
                            continue
                        yearly[yi] = float(v)
                    years_sorted = sorted(yearly.keys())
            except Exception:
                yearly = {}
                years_sorted = []
        else:
            # pandas not available, use csv module and try to parse simple two-column year,value
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    # try to find year and value columns
                    year_idx = None
                    val_idx = None
                    if header:
                        for i, c in enumerate(header):
                            lc = str(c).lower()
                            if 'year' in lc or '年' in lc:
                                year_idx = i
                            if 'value' in lc or '數值' in lc or 'value' in lc:
                                val_idx = i
                    if year_idx is None or val_idx is None:
                        # fallback: assume first two columns are year and value
                        year_idx = 0
                        val_idx = 1
                    for row in reader:
                        try:
                            yi = int(row[year_idx])
                            v = float(row[val_idx])
                        except Exception:
                            continue
                        yearly[yi] = yearly.get(yi, 0.0) + v
                    years_sorted = sorted(yearly.keys())
            except Exception:
                yearly = {}
                years_sorted = []
    year_idx = 0
    last_switch = time.time()
    switch_interval = 3
    transition_duration = 1.2  # 秒
    transition_start = None
    transition_ratio = 1.0
    # 归一化
    if years_sorted:
        vals = [yearly[y] for y in years_sorted]
        minv, maxv = min(vals), max(vals)
    else:
        minv, maxv = 0, 1

    # 初始点云
    if years_sorted:
        cur_year = years_sorted[year_idx]
        gsr_value = yearly[cur_year]
        norm = (gsr_value - minv) / (maxv - minv) if maxv > minv else 0.5
        multiplier = 0.8 + norm * 2.2
    else:
        multiplier = 1.0
    points_a = generate_point_cloud(center_x, center_y, num_points=480, std_x=80, std_y=60, multiplier=multiplier)
    points_b = points_a.copy()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))
        t = time.time()

        # draw contour lines driven by the current point cloud, smoothly blended between a and b
        draw_contour_lines(screen, width, height, center_x, center_y, layers=22, points_a=points_a, points_b=points_b, blend=transition_ratio)

        # 年份切换与点云平滑过渡
        if years_sorted:
            if t - last_switch > switch_interval:
                year_idx = (year_idx + 1) % len(years_sorted)
                last_switch = t
                transition_start = t
                # 生成新点云
                cur_year = years_sorted[year_idx]
                gsr_value = yearly[cur_year]
                norm = (gsr_value - minv) / (maxv - minv) if maxv > minv else 0.5
                multiplier = 0.8 + norm * 2.2
                points_b = generate_point_cloud(center_x, center_y, num_points=480, std_x=80, std_y=60, multiplier=multiplier)
                transition_ratio = 0.0
            # 过渡动画
            if transition_start is not None:
                transition_ratio = min(1.0, (t - transition_start) / transition_duration)
                if transition_ratio >= 1.0:
                    points_a = points_b
                    transition_start = None
            year_text_str = f"Year: {years_sorted[year_idx]}   Total: {yearly[years_sorted[year_idx]]:.1f}"
        else:
            year_text_str = "Year: #   Total: -"
            transition_ratio = 1.0

        draw_point_cloud(screen, points_a, points_b, t, transition_ratio)

        font = pygame.font.SysFont(None, 64)
        txt = font.render("Total Solar Radiation", True, (240, 80, 150))
        screen.blit(txt, (60, 60))
        font2 = pygame.font.SysFont(None, 36)
        txt2 = font2.render("visualization", True, (80, 120, 220))
        screen.blit(txt2, (60, 120))
        font3 = pygame.font.SysFont(None, 36)
        yt = font3.render(year_text_str, True, (50, 50, 60))
        screen.blit(yt, (60, 180))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    main_visualization()
# This code is part of a visualization project and should be run in an environment with pygame and pandas installed.
