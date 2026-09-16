import os
import glob
import cv2
import numpy as np

from gui import launch_gui
from background_subtraction import SelectiveDepthBackgroundSubtractor, DepthBlobSegmenter


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


def create_frame_stamp_bar(width: int, current_frame: int, total_frames: int, is_paused: bool) -> np.ndarray:
    """Generates the bottom status banner showing frame index and player status."""
    banner_height = 40
    banner = np.zeros((banner_height, width, 3), dtype=np.uint8)

    status_str = "PAUSED (Click/Space to Play)" if is_paused else "PLAYING (Click/Space to Pause)"
    stamp_text = f"Frame: {current_frame + 1} / {total_frames}  |  Status: {status_str}"
    controls_text = "[SPACE/CLICK]: Play/Pause   |   [Q/ESC]: Quit"

    cv2.putText(
        banner, stamp_text, (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA
    )

    cv2.putText(
        banner, controls_text, (max(width - 430, 200), 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA
    )

    return banner


def main():
    config = launch_gui()
    if not config:
        print("Pipeline execution canceled.")
        return

    data_dir = config["data_dir"]
    alpha = config["alpha"]
    delta = config["delta"]
    sobel_thresh = config["sobel_thresh"]

    depth_dir = os.path.join(data_dir, "depth")
    rgb_dir = os.path.join(data_dir, "RGB")

    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.*")))
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.*")))

    total_frames = len(depth_files)
    if total_frames == 0:
        print(f"No depth images found in {depth_dir}")
        return

    subtractor = SelectiveDepthBackgroundSubtractor(alpha=alpha, delta=delta)  #[cite: 1]
    segmenter = DepthBlobSegmenter(sobel_thresh=sobel_thresh, min_area=300)  #[cite: 1]

    processed_cache = {}
    last_processed_index = -1

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

    cv2.createTrackbar("Scrubber", window_name, 0, total_frames - 1, on_trackbar)
    cv2.setMouseCallback(window_name, on_mouse_click)

    def process_frame(idx: int):
        depth_frame = cv2.imread(depth_files[idx], cv2.IMREAD_UNCHANGED)
        rgb_frame = cv2.imread(rgb_files[idx])

        if depth_frame is None or rgb_frame is None:
            return None

        fg_mask = subtractor.apply(depth_frame)  #[cite: 1]
        clean_blobs_mask, blobs, edge_mask = segmenter.process(depth_frame, fg_mask)  #[cite: 1]

        colored_blobs = np.zeros_like(rgb_frame)
        for blob in blobs:
            color = np.random.randint(50, 255, size=3).tolist()
            colored_blobs[blob['mask']] = color
            x, y, w, h = blob['bbox']
            cv2.rectangle(colored_blobs, (x, y), (x + w, y + h), color, 1)

        fg_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        edge_bgr = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)

        panel1 = add_panel_label(rgb_frame, "1. Input RGB Stream")
        panel2 = add_panel_label(fg_bgr, "2. Module 1: Selective Background Subtraction")  #[cite: 1]
        panel3 = add_panel_label(edge_bgr, "3. Module 2: Depth Edge Detection (Sobel)")  #[cite: 1]
        panel4 = add_panel_label(colored_blobs, "4. Module 2: Edge-Split Blobs")  #[cite: 1]

        top_row = np.hstack((panel1, panel2))
        bottom_row = np.hstack((panel3, panel4))
        return np.vstack((top_row, bottom_row))

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

        composite_grid = processed_cache[current_frame_idx]

        if composite_grid is not None:
            banner = create_frame_stamp_bar(
                width=composite_grid.shape[1],
                current_frame=current_frame_idx,
                total_frames=total_frames,
                is_paused=is_paused
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