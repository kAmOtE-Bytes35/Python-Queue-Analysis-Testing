import os
import re
import glob
import colorsys
import cv2
import numpy as np

from gui import launch_gui
from background_subtraction import (
    SelectiveDepthBackgroundSubtractor,
    DepthBlobSegmenter,
    HeadLocalizer,
)
from geometry import CameraGeometry, RansacPlaneFitter


def color_for_head_index(index: int):
    golden_ratio_conjugate = 0.618033988749895
    hue = (index * golden_ratio_conjugate) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def add_panel_label(img: np.ndarray, text: str) -> np.ndarray:
    output = img.copy()
    h, w = output.shape[:2]
    banner_height = 30
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
    cv2.putText(
        output, text, (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )
    return output


def render_ransac_vis_panel(depth_image: np.ndarray, plane: np.ndarray, inliers: np.ndarray, floor_mask: np.ndarray) -> np.ndarray:
    """Generates a visualization panel showing RANSAC inliers on base frame 0.png."""
    h, w = depth_image.shape
    depth_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    vis_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

    if inliers is not None and floor_mask is not None:
        # Reconstruct 2D inlier mask
        inlier_img_mask = np.zeros((h, w), dtype=bool)
        inlier_img_mask[floor_mask] = inliers

        # Highlight ground plane inliers in green
        green_overlay = vis_bgr.copy()
        green_overlay[inlier_img_mask] = (0, 255, 0)
        cv2.addWeighted(green_overlay, 0.45, vis_bgr, 0.55, 0, vis_bgr)

    a, b, c, d = plane
    inlier_pct = (np.sum(inliers) / len(inliers) * 100) if inliers is not None else 0.0

    # Draw Calibration Specs Overlay
    info_bg = vis_bgr.copy()
    cv2.rectangle(info_bg, (15, 40), (w - 15, 140), (0, 0, 0), -1)
    cv2.addWeighted(info_bg, 0.7, vis_bgr, 0.3, 0, vis_bgr)

    cv2.putText(vis_bgr, f"RANSAC Plane Equation: {a:.4f}X + {b:.4f}Y + {c:.4f}Z + {d:.1f} = 0", 
                (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis_bgr, f"Normal Vector (n): [{a:.3f}, {b:.3f}, {c:.3f}]", 
                (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis_bgr, f"Floor Inlier Points: {np.sum(inliers) if inliers is not None else 0} ({inlier_pct:.1f}%)", 
                (25, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

    return add_panel_label(vis_bgr, "7. RANSAC Ground Plane Fit (0.png Inliers)")


def render_coordinate_map_panel(heads_3d: list, target_shape=(480, 640, 3)) -> np.ndarray:
    """Renders a 2D top-down Bird's-Eye View map showing real-life (X, Z) positions in meters."""
    h, w, _ = target_shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :] = (25, 25, 25)  # Dark gray grid background

    # Map bounds in meters
    x_min, x_max = -3.0, 3.0
    z_min, z_max = 0.0, 6.0

    scale_x = w / (x_max - x_min)
    scale_z = h / (z_max - z_min)

    def world_to_map(x_m, z_m):
        px = int((x_m - x_min) * scale_x)
        py = int(h - (z_m - z_min) * scale_z)
        return px, py

    # Draw Grid Lines (every 1 meter)
    for zm in range(int(z_min), int(z_max) + 1):
        _, py = world_to_map(0, zm)
        cv2.line(canvas, (0, py), (w, py), (45, 45, 45), 1)
        cv2.putText(canvas, f"{zm}m", (10, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)

    for xm in range(int(x_min), int(x_max) + 1):
        px, _ = world_to_map(xm, 0)
        cv2.line(canvas, (px, 0), (px, h), (45, 45, 45), 1)
        cv2.putText(canvas, f"{xm}m", (px + 4, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)

    # Draw Camera Position (Origin)
    cam_x, cam_y = world_to_map(0, 0)
    cv2.circle(canvas, (cam_x, cam_y), 8, (0, 0, 255), -1)
    cv2.putText(canvas, "Camera (0,0)", (cam_x - 35, cam_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    # Plot Heads on 2D Map
    for head in heads_3d:
        hid = head['head_id']
        x_m = head['X_m']
        z_m = head['Z_m']
        h_m = head['H_m']
        color = color_for_head_index(hid)

        px, py = world_to_map(x_m, z_m)
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(canvas, (px, py), 7, color, -1)
            cv2.circle(canvas, (px, py), 9, (255, 255, 255), 1)
            label = f"H{hid} ({x_m:+.2f}m, {z_m:.2f}m)"
            cv2.putText(canvas, label, (px + 10, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return add_panel_label(canvas, f"8. Real-Life 2D Map [Bird's-Eye View] (count={len(heads_3d)})")


def render_tab_bar(width: int, active_tab: int) -> np.ndarray:
    """Renders top navigation tab bar for switching views."""
    tab_height = 35
    bar = np.zeros((tab_height, width, 3), dtype=np.uint8)
    tabs = ["1: Processing Pipeline", "2: Ground Plane Calibration", "3: 2D Head Coordinate Map", "4: 8-Panel Dashboard Grid"]
    
    tab_w = width // len(tabs)
    for i, title in enumerate(tabs):
        x1 = i * tab_w
        x2 = (i + 1) * tab_w
        is_active = (i + 1) == active_tab
        bg_color = (60, 60, 60) if is_active else (25, 25, 25)
        text_color = (0, 255, 255) if is_active else (180, 180, 180)
        
        cv2.rectangle(bar, (x1, 0), (x2, tab_height), bg_color, -1)
        cv2.rectangle(bar, (x1, 0), (x2, tab_height), (80, 80, 80), 1)
        cv2.putText(bar, title, (x1 + 15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1 if not is_active else 2, cv2.LINE_AA)

    return bar


def create_frame_stamp_bar(width: int, position: int, total_frames: int,
                            frame_number: int, is_paused: bool, head_count: int) -> np.ndarray:
    banner_height = 40
    banner = np.zeros((banner_height, width, 3), dtype=np.uint8)
    status_str = "PAUSED (Click/Space to Play)" if is_paused else "PLAYING (Click/Space to Pause)"
    stamp_text = (
        f"Pos: {position + 1}/{total_frames} | Frame #: {frame_number} | "
        f"Heads: {head_count} | Status: {status_str}"
    )
    controls_text = "Tabs: [1-4] or Click Top Bar | [SPACE]: Pause | [Q/ESC]: Quit"
    cv2.putText(banner, stamp_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, controls_text, (max(width - 480, 200), 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    return banner


def _extract_frame_number(path: str):
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r'\d+', stem)
    return int(match.group()) if match else None


def _index_directory_by_frame_number(dir_path: str) -> dict:
    mapping = {}
    for path in glob.glob(os.path.join(dir_path, "*.*")):
        frame_num = _extract_frame_number(path)
        if frame_num is not None and frame_num not in mapping:
            mapping[frame_num] = path
    return mapping


def build_frame_pairs(data_dir: str):
    depth_dir = os.path.join(data_dir, "depth")
    rgb_dir = os.path.join(data_dir, "RGB")
    depth_map = _index_directory_by_frame_number(depth_dir)
    rgb_map = _index_directory_by_frame_number(rgb_dir)
    common = sorted(set(depth_map) & set(rgb_map))
    depth_files = [depth_map[n] for n in common]
    rgb_files = [rgb_map[n] for n in common]
    return common, depth_files, rgb_files


def calibrate_ground_plane(data_dir: str):
    """Extracts ground plane and returns plane parameters along with calibration visualization data."""
    base_frame_path = os.path.join(data_dir, "depth", "0.png")
    depth_image = cv2.imread(base_frame_path, cv2.IMREAD_UNCHANGED)
    
    geom = CameraGeometry()
    if depth_image is None:
        print("[WARN] Base frame 0.png not found. Falling back to baseline plane.")
        return geom.baseline_plane, None, None, None

    print("[INFO] Running RANSAC ground plane calibration on 0.png...")
    fitter = RansacPlaneFitter(distance_threshold=30.0, max_iterations=2000)
    h, w = depth_image.shape
    floor_roi_mask = np.zeros_like(depth_image, dtype=bool)
    floor_roi_mask[int(h*0.4):, :] = (depth_image[int(h*0.4):, :] > 0)
    
    point_cloud = geom.backproject_point_cloud(depth_image, mask=floor_roi_mask)
    fitted_plane, inliers = fitter.fit(point_cloud)
    
    if fitted_plane is not None:
        print(f"[INFO] Ground plane calibrated: {fitted_plane}")
        return fitted_plane, depth_image, inliers, floor_roi_mask
    else:
        print("[WARN] RANSAC failed. Falling back to baseline plane.")
        return geom.baseline_plane, depth_image, None, floor_roi_mask


def main():
    config = launch_gui()
    if not config:
        return

    data_dir = config["data_dir"]
    alpha = config["alpha"]
    delta = config["delta"]
    sobel_thresh = config["sobel_thresh"]
    epsilon = config["epsilon"]

    # --- 1. Calibrate Ground Plane ---
    ground_plane, base_depth_img, ransac_inliers, floor_mask = calibrate_ground_plane(data_dir)
    camera_geom = CameraGeometry()

    # Pre-render RANSAC Calibration Panel
    ransac_panel = render_ransac_vis_panel(base_depth_img, ground_plane, ransac_inliers, floor_mask)

    frame_numbers, depth_files, rgb_files = build_frame_pairs(data_dir)
    total_frames = len(frame_numbers)
    if total_frames == 0:
        return

    subtractor = SelectiveDepthBackgroundSubtractor(alpha=alpha, delta=delta)
    segmenter = DepthBlobSegmenter(sobel_thresh=sobel_thresh, min_area=300)
    head_localizer = HeadLocalizer(epsilon=epsilon, min_head_area=20)

    processed_cache = {}
    last_processed_index = -1
    last_good_panel_shape = None

    window_name = "Depth Processing Pipeline Player"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 780)

    current_frame_idx = 0
    active_tab = 1
    is_paused = False
    slider_user_seeking = False

    def on_trackbar(val):
        nonlocal current_frame_idx, slider_user_seeking
        current_frame_idx = val
        slider_user_seeking = True

    def on_mouse_click(event, x, y, flags, param):
        nonlocal is_paused, active_tab
        if event == cv2.EVENT_LBUTTONDOWN:
            if y < 35:  # Click on Tab Bar
                win_w = cv2.getWindowImageRect(window_name)[2]
                tab_w = max(win_w // 4, 1)
                active_tab = min(max(1, (x // tab_w) + 1), 4)
            else:
                is_paused = not is_paused

    cv2.createTrackbar("Scrubber", window_name, 0, max(total_frames - 1, 0), on_trackbar)
    cv2.setMouseCallback(window_name, on_mouse_click)

    def process_frame(idx: int):
        nonlocal last_good_panel_shape
        frame_number = frame_numbers[idx]
        depth_frame = cv2.imread(depth_files[idx], cv2.IMREAD_UNCHANGED)
        rgb_frame = cv2.imread(rgb_files[idx])

        if depth_frame is None or rgb_frame is None:
            fallback = last_good_panel_shape if last_good_panel_shape else (480, 640, 3)
            err = np.zeros(fallback, dtype=np.uint8)
            err[:, :] = (0, 0, 80)
            err = add_panel_label(err, f"READ ERROR - frame #{frame_number}")
            return {"panels": [err]*8, "count": 0}

        last_good_panel_shape = rgb_frame.shape

        fg_mask = subtractor.apply(depth_frame)
        _, blobs, edge_mask = segmenter.process(depth_frame, fg_mask)
        _, heads = head_localizer.process(depth_frame, blobs)

        colored_blobs = np.zeros_like(rgb_frame)
        for blob in blobs:
            color = np.random.randint(50, 255, size=3).tolist()
            colored_blobs[blob['mask']] = color
            x, y, w, h = blob['bbox']
            cv2.rectangle(colored_blobs, (x, y), (x + w, y + h), color, 1)

        colored_heads = np.zeros_like(rgb_frame)
        heads_on_rgb = rgb_frame.copy()
        heads_3d_data = []

        for head in heads:
            color = color_for_head_index(head['head_id'])
            u, v = head['top_point']
            d_hat = head['depth_min']

            # Backproject to 3D Camera Coordinates (in mm)
            point_3d = camera_geom.backproject_pixel(u, v, d_hat)
            height_mm = camera_geom.distance_to_plane(point_3d, ground_plane)
            
            x_m = point_3d[0] / 1000.0
            z_m = point_3d[2] / 1000.0
            height_m = height_mm / 1000.0

            heads_3d_data.append({
                'head_id': head['head_id'],
                'X_m': x_m,
                'Z_m': z_m,
                'H_m': height_m
            })

            colored_heads[head['mask']] = color
            hx, hy, hw, hh = head['bbox']
            cv2.rectangle(colored_heads, (hx, hy), (hx + hw, hy + hh), color, 1)
            cv2.circle(colored_heads, (u, v), 3, (0, 0, 255), -1)

            cv2.rectangle(heads_on_rgb, (hx, hy), (hx + hw, hy + hh), color, 2)
            cv2.circle(heads_on_rgb, (u, v), 4, (0, 0, 255), -1)

            label_p5 = f"H{head['head_id']}" + ("*" if head['split_from_shared_blob'] else "")
            label_p6 = f"H{head['head_id']} ({height_m:.2f}m)"

            cv2.putText(colored_heads, label_p5, (hx, max(hy - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(heads_on_rgb, label_p6, (hx, max(hy - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        fg_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        edge_bgr = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)

        p1 = add_panel_label(rgb_frame, f"1. Input RGB Stream (frame #{frame_number})")
        p2 = add_panel_label(fg_bgr, "2. Background Subtraction")
        p3 = add_panel_label(edge_bgr, "3. Depth Edge Detection (Sobel)")
        p4 = add_panel_label(colored_blobs, "4. Edge-Split Blobs")
        p5 = add_panel_label(colored_heads, "5. Head Localization")
        p6 = add_panel_label(heads_on_rgb, f"6. Height Est. (count={len(heads)})")
        p7 = ransac_panel
        p8 = render_coordinate_map_panel(heads_3d_data, target_shape=rgb_frame.shape)

        return {"panels": [p1, p2, p3, p4, p5, p6, p7, p8], "count": len(heads)}

    while True:
        if current_frame_idx > last_processed_index:
            for f in range(last_processed_index + 1, current_frame_idx + 1):
                processed_cache[f] = process_frame(f)
                last_processed_index = f
        elif current_frame_idx not in processed_cache:
            subtractor = SelectiveDepthBackgroundSubtractor(alpha=alpha, delta=delta)
            last_processed_index = -1
            for f in range(0, current_frame_idx + 1):
                processed_cache[f] = process_frame(f)
                last_processed_index = f

        frame_data = processed_cache[current_frame_idx]
        panels = frame_data["panels"]
        head_count = frame_data["count"]

        # Build active tab layout
        if active_tab == 1:
            top_row = np.hstack((panels[0], panels[1], panels[2]))
            bot_row = np.hstack((panels[3], panels[4], panels[5]))
            main_display = np.vstack((top_row, bot_row))
        elif active_tab == 2:
            main_display = cv2.resize(panels[6], (panels[0].shape[1]*2, panels[0].shape[0]*2))
        elif active_tab == 3:
            main_display = cv2.resize(panels[7], (panels[0].shape[1]*2, panels[0].shape[0]*2))
        else:  # Tab 4: 8-Panel Dashboard Grid
            r1 = np.hstack((panels[0], panels[1], panels[2], panels[3]))
            r2 = np.hstack((panels[4], panels[5], panels[6], panels[7]))
            main_display = np.vstack((r1, r2))

        tab_bar = render_tab_bar(main_display.shape[1], active_tab)
        banner = create_frame_stamp_bar(
            width=main_display.shape[1], position=current_frame_idx,
            total_frames=total_frames, frame_number=frame_numbers[current_frame_idx],
            is_paused=is_paused, head_count=head_count,
        )

        full_window = np.vstack((tab_bar, main_display, banner))

        if not slider_user_seeking:
            cv2.setTrackbarPos("Scrubber", window_name, current_frame_idx)
        slider_user_seeking = False

        cv2.imshow(window_name, full_window)

        key = cv2.waitKey(30 if not is_paused else 100) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == 32:
            is_paused = not is_paused
        elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
            active_tab = int(chr(key))

        if not is_paused:
            if current_frame_idx < total_frames - 1:
                current_frame_idx += 1
            else:
                is_paused = True

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()