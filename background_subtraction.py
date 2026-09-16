import cv2
import numpy as np

class SelectiveDepthBackgroundSubtractor:
    def __init__(self, alpha: float = 0.01, delta: float = 200.0):
        """
        Selective Background Subtractor following Bondi et al. (AVSS 2014)[cite: 1]
        """
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.background = None

    def apply(self, depth_frame: np.ndarray) -> np.ndarray:
        frame_float = depth_frame.astype(np.float32)

        if self.background is None:
            self.background = frame_float.copy()
            return np.zeros(depth_frame.shape, dtype=np.uint8)

        # 1. Depth difference |B_t(x,y) - F_t(x,y)|[cite: 1]
        diff = np.abs(self.background - frame_float)
        foreground_mask = (diff > self.delta) & (frame_float > 0)

        # 2. Update background model selectively using binary indicator delta_t[cite: 1]
        delta_t = (~foreground_mask).astype(np.float32)
        update_weight = self.alpha * delta_t
        self.background = (update_weight * frame_float) + ((1.0 - update_weight) * self.background)

        return (foreground_mask * 255).astype(np.uint8)


class DepthBlobSegmenter:
    def __init__(self, sobel_thresh: float = 80.0, min_area: int = 300):
        """
        Edge-guided blob splitting using Sobel gradient on depth maps[cite: 1].
        """
        self.sobel_thresh = float(sobel_thresh)
        self.min_area = int(min_area)

    def process(self, depth_frame: np.ndarray, fg_mask: np.ndarray):
        depth_float = depth_frame.astype(np.float32)

        # 1. Compute depth edges using Sobel operator[cite: 1]
        grad_x = cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)

        # 2. Binarize and dilate depth boundary edges
        edge_mask = (magnitude > self.sobel_thresh).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edge_mask_dilated = cv2.dilate(edge_mask, kernel, iterations=1)

        # 3. Mask foreground with depth edges to split connected people[cite: 1]
        separated_fg = cv2.bitwise_and(fg_mask, cv2.bitwise_not(edge_mask_dilated))

        # 4. Connected components analysis and minimum area filtering[cite: 1]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(separated_fg, connectivity=8)

        clean_mask = np.zeros_like(separated_fg)
        retained_blobs = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area:  # Discard blobs smaller than 300 pixels[cite: 1]
                blob_pixels_mask = (labels == i)
                clean_mask[blob_pixels_mask] = 255
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                retained_blobs.append({
                    'id': i,
                    'area': area,
                    'bbox': (x, y, w, h),
                    'mask': blob_pixels_mask
                })

        return clean_mask, retained_blobs, edge_mask