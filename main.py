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


def color_for_head_index(index: int):
    """
    Deterministic, visually distinct BGR color for a given head index,
    using golden-ratio hue stepping so consecutive indices land far apart
    on the color wheel. Colors are assigned PER HEAD, not per parent blob,
    so two heads recovered from a single merged blob get different colors
    -- this is what makes a blob split visible in the GUI.
    """
    golden_ratio_conjugate = 0.618033988749895
    hue = (index * golden_ratio_conjugate) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))  # OpenCV uses BGR


def add_panel_label(img: np.ndarray, text: str) -> np.ndarray:
    """Overlay dark header banner with label text over a panel."""
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


def create_frame_stamp_bar(width: int, position: int, total_frames: int,
                            frame_number: int, is_paused: bool, head_count: int) -> np.ndarray:
    """Generates the bottom status banner: playback position + actual dataset
    frame number (these can differ if some frame numbers are missing on disk)
    + live head-detection count[cite: 1]."""
    banner_height = 40
    banner = np.zeros((banner_height, width, 3), dtype=np.uint8)

    status_str = "PAUSED (Click/Space to Play)" if is_paused else "PLAYING (Click/Space to Pause)"
    stamp_text = (
        f"Position: {position + 1}/{total_frames}  |  Dataset Frame #: {frame_number}  |  "
        f"Heads Detected: {head_count}  |  Status: {status_str}"
    )
    controls_text = "[SPACE/CLICK]: Play/Pause   |   [Q/ESC]: Quit"

    cv2.putText(
        banner, stamp_text, (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
    )

    cv2.putText(
        banner, controls_text, (max(width - 430, 200), 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA
    )

    return banner


def _extract_frame_number(path: str):
    """Pull the leading integer out of a filename stem, e.g. '104.png' -> 104."""
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r'\d+', stem)
    return int(match.group()) if match else None


def _index_directory_by_frame_number(dir_path: str) -> dict:
    """
    Build {frame_number: filepath} for every file in dir_path, using the
    NUMERIC value embedded in the filename rather than filesystem/string
    order. This is required because 'sorted(glob.glob(...))' on filenames
    like '0.png', '1.png', ..., '917.png' performs a LEXICOGRAPHIC sort
    (e.g. '10.png' sorts before '2.png'), silently misaligning every frame
    past the single digits with the wrong file.
    """
    mapping = {}
    duplicates = []
    for path in glob.glob(os.path.join(dir_path, "*.*")):
        frame_num = _extract_frame_number(path)
        if frame_num is None:
            continue
        if frame_num in mapping:
            duplicates.append((frame_num, mapping[frame_num], path))
            continue
        mapping[frame_num] = path

    if duplicates:
        print(f"[WARN] {len(duplicates)} duplicate frame numbers found in {dir_path} "
              f"(kept the first, ignored the rest): "
              f"{[d[0] for d in duplicates[:20]]}{' ...' if len(duplicates) > 20 else ''}")

    return mapping


def build_frame_pairs(data_dir: str):
    """
    Pairs RGB and depth files by their actual numeric frame number (not by
    position in a sorted() file list), so:
      1. Numeric order is used instead of lexicographic order (fixes frames
         being loaded out of sequence, e.g. '11.png' shown as "frame 13").
      2. If a frame number exists in one folder but not the other, it is
         reported and SKIPPED rather than silently shifting every
         subsequent frame by one position (which produces the periodic
         stutter/repeat symptom, since a positionally-misaligned read
         failure leaves the previous frame frozen on screen).
    Returns a sorted list of frame numbers and the matching file lists.
    """
    depth_dir = os.path.join(data_dir, "depth")
    rgb_dir = os.path.join(data_dir, "RGB")

    depth_map = _index_directory_by_frame_number(depth_dir)
    rgb_map = _index_directory_by_frame_number(rgb_dir)

    depth_only = sorted(set(depth_map) - set(rgb_map))
    rgb_only = sorted(set(rgb_map) - set(depth_map))
    common = sorted(set(depth_map) & set(rgb_map))

    if depth_only:
        print(f"[WARN] {len(depth_only)} depth frame(s) have NO matching RGB file, "
              f"skipped: {depth_only[:30]}{' ...' if len(depth_only) > 30 else ''}")
    if rgb_only:
        print(f"[WARN] {len(rgb_only)} RGB frame(s) have NO matching depth file, "
              f"skipped: {rgb_only[:30]}{' ...' if len(rgb_only) > 30 else ''}")

    # Sanity check: flag any gaps in the numbering itself (e.g. frame 45 simply
    # missing from the whole dataset, not just mismatched between folders).
    if common:
        expected = set(range(common[0], common[-1] + 1))
        gaps = sorted(expected - set(common))
        if gaps:
            print(f"[WARN] {len(gaps)} frame number(s) missing entirely from the dataset "
                  f"(gap in numbering): {gaps[:30]}{' ...' if len(gaps) > 30 else ''}")

    depth_files = [depth_map[n] for n in common]
    rgb_files = [rgb_map[n] for n in common]

    print(f"[INFO] Paired {len(common)} frames "
          f"(range {common[0] if common else '-'}..{common[-1] if common else '-'}) "
          f"in numeric order.")

    return common, depth_files, rgb_files


def main():
    config = launch_gui()
    if not config:
        print("Pipeline execution canceled.")
        return

    data_dir = config["data_dir"]
    alpha = config["alpha"]
    delta = config["delta"]
    sobel_thresh = config["sobel_thresh"]
    epsilon = config["epsilon"]

    frame_numbers, depth_files, rgb_files = build_frame_pairs(data_dir)

    total_frames = len(frame_numbers)
    if total_frames == 0:
        print(f"No matching depth/RGB frame pairs found under {data_dir}")
        return

    subtractor = SelectiveDepthBackgroundSubtractor(alpha=alpha, delta=delta)  #[cite: 1]
    segmenter = DepthBlobSegmenter(sobel_thresh=sobel_thresh, min_area=300)  #[cite: 1]
    head_localizer = HeadLocalizer(epsilon=epsilon, min_head_area=20)  #[cite: 1]

    processed_cache = {}
    last_processed_index = -1
    last_good_panel_shape = None  # (h, w, 3) of the most recent successfully read frame

    # Resizable Window Setup
    window_name = "Depth Processing Pipeline Player"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Enables free resizing
    cv2.resizeWindow(window_name, 1280, 720)        # Default starting resolution

    current_frame_idx = 0
    is_paused = False
    slider_user_seeking = False

    def on_trackbar(val):
        nonlocal current_frame_idx, slider_user_seeking
        current_frame_idx = val
        slider_user_seeking = True

    def on_mouse_click(event, x, y, flags, param):
        nonlocal is_paused
        # Toggle pause/resume on left mouse click
        if event == cv2.EVENT_LBUTTONDOWN:
            is_paused = not is_paused

    cv2.createTrackbar("Scrubber", window_name, 0, max(total_frames - 1, 0), on_trackbar)
    cv2.setMouseCallback(window_name, on_mouse_click)

    def make_error_panel(shape, message: str) -> np.ndarray:
        h, w = shape[0], shape[1]
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        panel[:, :] = (0, 0, 80)  # dark red so a bad frame is unmistakable, not a frozen repeat
        cv2.putText(
            panel, message, (20, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )
        return panel

    def process_frame(idx: int):
        nonlocal last_good_panel_shape

        frame_number = frame_numbers[idx]
        depth_path = depth_files[idx]
        rgb_path = rgb_files[idx]

        depth_frame = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        rgb_frame = cv2.imread(rgb_path)

        if depth_frame is None or rgb_frame is None:
            # Instead of returning None (which used to leave the previous
            # frame frozen on screen -- the "stutter" symptom), render a
            # clearly-marked error panel so bad reads are visible, not silent.
            print(f"[ERROR] Failed to read frame #{frame_number} "
                  f"(depth='{depth_path}', rgb='{rgb_path}'). Substituting an error panel.")
            fallback_shape = last_good_panel_shape if last_good_panel_shape else (480, 640, 3)
            panel_h, panel_w = fallback_shape[0], fallback_shape[1]
            err = make_error_panel(
                (panel_h, panel_w),
                f"FRAME READ ERROR (dataset frame #{frame_number})"
            )
            err = add_panel_label(err, f"READ ERROR - frame #{frame_number}")
            grid_row = np.hstack((err, err, err))
            return np.vstack((grid_row, grid_row)), 0

        last_good_panel_shape = rgb_frame.shape

        fg_mask = subtractor.apply(depth_frame)  #[cite: 1]
        clean_blobs_mask, blobs, edge_mask = segmenter.process(depth_frame, fg_mask)  #[cite: 1]
        head_mask, heads = head_localizer.process(depth_frame, blobs)  #[cite: 1]

        colored_blobs = np.zeros_like(rgb_frame)
        blob_colors = {}
        for blob in blobs:
            color = np.random.randint(50, 255, size=3).tolist()
            blob_colors[blob['id']] = color
            colored_blobs[blob['mask']] = color
            x, y, w, h = blob['bbox']
            cv2.rectangle(colored_blobs, (x, y), (x + w, y + h), color, 1)

        # Panel 5: isolated head regions. Each head gets its OWN color
        # (color_for_head_index), independent of its parent blob's color --
        # so if one blob was split into two heads by HeadLocalizer, those
        # two heads render in visibly different colors instead of both
        # inheriting the single blob color. A faint gray outline of each
        # parent blob is drawn first so you can still see which heads came
        # from the same original (possibly merged) blob[cite: 1].
        colored_heads = np.zeros_like(rgb_frame)
        for blob in blobs:
            x, y, w, h = blob['bbox']
            cv2.rectangle(colored_heads, (x, y), (x + w, y + h), (90, 90, 90), 1)

        for head in heads:
            color = color_for_head_index(head['head_id'])
            colored_heads[head['mask']] = color
            x, y, w, h = head['bbox']
            cv2.rectangle(colored_heads, (x, y), (x + w, y + h), color, 1)
            cv2.circle(colored_heads, head['top_point'], 3, (0, 0, 255), -1)
            label = f"H{head['head_id']}" + ("*" if head['split_from_shared_blob'] else "")
            cv2.putText(colored_heads, label, (x, max(y - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Panel 6: head detections overlaid on the RGB frame, same
        # per-head coloring as panel 5[cite: 1]
        heads_on_rgb = rgb_frame.copy()
        for head in heads:
            color = color_for_head_index(head['head_id'])
            x, y, w, h = head['bbox']
            cv2.rectangle(heads_on_rgb, (x, y), (x + w, y + h), color, 2)
            cv2.circle(heads_on_rgb, head['top_point'], 4, (0, 0, 255), -1)
            label = f"H{head['head_id']}" + ("*" if head['split_from_shared_blob'] else "")
            cv2.putText(heads_on_rgb, label, (x, max(y - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        fg_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        edge_bgr = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)

        panel1 = add_panel_label(rgb_frame, f"1. Input RGB Stream (frame #{frame_number})")
        panel2 = add_panel_label(fg_bgr, "2. Module 1: Selective Background Subtraction")  #[cite: 1]
        panel3 = add_panel_label(edge_bgr, "3. Module 2: Depth Edge Detection (Sobel)")  #[cite: 1]
        panel4 = add_panel_label(colored_blobs, "4. Module 2: Edge-Split Blobs")  #[cite: 1]
        panel5 = add_panel_label(colored_heads, "5. Module 3: Head Localization (split-aware, *=blob split)")  #[cite: 1]
        panel6 = add_panel_label(heads_on_rgb, f"6. Heads on RGB  (count={len(heads)})")  #[cite: 1]

        top_row = np.hstack((panel1, panel2, panel3))
        bottom_row = np.hstack((panel4, panel5, panel6))
        return np.vstack((top_row, bottom_row)), len(heads)

    while True:
        # State alignment for forward/backward scrubbing
        if current_frame_idx > last_processed_index:
            for f in range(last_processed_index + 1, current_frame_idx + 1):
                processed_cache[f] = process_frame(f)
                last_processed_index = f
        elif current_frame_idx not in processed_cache:
            subtractor = SelectiveDepthBackgroundSubtractor(alpha=alpha, delta=delta)  #[cite: 1]
            last_processed_index = -1
            for f in range(0, current_frame_idx + 1):
                processed_cache[f] = process_frame(f)
                last_processed_index = f

        composite_grid, head_count = processed_cache[current_frame_idx]

        if composite_grid is not None:
            banner = create_frame_stamp_bar(
                width=composite_grid.shape[1],
                position=current_frame_idx,
                total_frames=total_frames,
                frame_number=frame_numbers[current_frame_idx],
                is_paused=is_paused,
                head_count=head_count,
            )
            full_frame = np.vstack((composite_grid, banner))

            if not slider_user_seeking:
                cv2.setTrackbarPos("Scrubber", window_name, current_frame_idx)
            slider_user_seeking = False

            cv2.imshow(window_name, full_frame)

        key = cv2.waitKey(30 if not is_paused else 100) & 0xFF

        if key in (ord('q'), 27):  # 'q' or ESC to exit
            break
        elif key == 32:  # Spacebar to pause/resume
            is_paused = not is_paused

        if not is_paused:
            if current_frame_idx < total_frames - 1:
                current_frame_idx += 1
            else:
                is_paused = True

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()