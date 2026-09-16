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


class HeadLocalizer:
    """
    Per-blob head extraction, following Section 3 of Bondi et al.[cite: 1],
    with an added spatial-connectivity pass so that two people who were
    fused into a single blob by the Sobel edge-splitting step still yield
    two separate head detections, provided their heads are not touching.

    For each blob B with pixel set P = {(x1,y1), ..., (xn,yn)}, the paper
    computes d_hat = min_p D(x,y) -- the closest (smallest depth) point in
    the blob -- then retains only pixels whose depth falls in the band
    [d_hat, d_hat + epsilon]. Because the head is the topmost, closest-to
    -camera part of a person in an overhead/angled RGB-D view, this band
    isolates the head region(s) and discards the rest of the body[cite: 1].

    IMPORTANT REFINEMENT: the paper's description implicitly assumes one
    head per blob, but a single blob can legitimately contain more than
    one head (e.g. two people standing close enough that the Sobel edge
    mask does not fully separate their bodies, while their heads are still
    spatially distinct at the top of the blob). Taking a single global
    min/max over the whole depth-band mask would silently merge two heads
    into one oversized bounding box. To avoid this, the depth-band mask
    for each blob is itself run through connected-component labeling, and
    each spatially-disconnected sub-region becomes its own head detection
    with its own bounding box.
    """

    def __init__(self, epsilon: float = 150.0, min_head_area: int = 20,
                 morph_kernel_size: int = 3):
        """
        Args:
            epsilon: depth band width (in the same units as the depth map,
                typically mm) added to the blob's minimum depth d_hat to
                define the retained range [d_hat, d_hat + epsilon][cite: 1].
            min_head_area: minimum pixel count for a candidate head region
                to be kept, filtering out noise fragments.
            morph_kernel_size: size of the morphological closing kernel
                applied to the depth-band mask before splitting it into
                sub-regions. This bridges small 1-2 pixel depth-noise gaps
                *within* a single real head so it isn't spuriously split
                into fragments, without being large enough to bridge the
                genuine gap between two separate people's heads. Set to 0
                or 1 to disable.
        """
        self.epsilon = float(epsilon)
        self.min_head_area = int(min_head_area)
        self.morph_kernel_size = int(morph_kernel_size)

    def process(self, depth_frame: np.ndarray, blobs: list):
        """
        Args:
            depth_frame: raw depth map (same frame passed to the segmenter).
            blobs: list of blob dicts produced by DepthBlobSegmenter.process,
                each containing at least 'id', 'mask', 'bbox'.

        Returns:
            head_mask: uint8 mask (255 at retained head pixels) for display,
                analogous to panel "Head detection" in Figure 1 of the paper.
            heads: list of dicts, one per surviving, spatially-disconnected
                head region, each with 'head_id' (globally unique within
                the frame), 'blob_id' (the parent blob it came from --
                multiple heads can share the same blob_id when a blob was
                split), 'split_from_shared_blob' (True if this head's
                parent blob yielded more than one head), 'mask', 'bbox',
                'centroid', 'top_point', 'depth_min', 'area'. 'top_point'
                is the highest (smallest image-y) pixel in the head region
                -- this is what Section 4 later back-projects onto the
                ground plane[cite: 1].
        """
        depth_float = depth_frame.astype(np.float32)
        head_mask = np.zeros(depth_frame.shape[:2], dtype=np.uint8)
        heads = []
        head_counter = 0

        use_morph = self.morph_kernel_size and self.morph_kernel_size > 1
        if use_morph:
            morph_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size)
            )

        for blob in blobs:
            blob_mask = blob['mask']

            # 1. d_hat = min_p D(x, y) over the blob's pixels[cite: 1].
            #    Ignore zero/invalid depth (sensor holes) when finding the min.
            blob_depths = depth_float[blob_mask]
            valid_depths = blob_depths[blob_depths > 0]
            if valid_depths.size == 0:
                continue
            d_hat = float(valid_depths.min())

            # 2. Retain pixels (x, y) of the blob with D(x, y) in
            #    [d_hat, d_hat + epsilon][cite: 1].
            in_band = (depth_float >= d_hat) & (depth_float <= d_hat + self.epsilon)
            band_mask = (in_band & blob_mask).astype(np.uint8) * 255

            if band_mask.max() == 0:
                continue

            # Bridge tiny depth-noise gaps within a single head before
            # splitting, so we don't over-fragment one real head.
            if use_morph:
                band_mask = cv2.morphologyEx(band_mask, cv2.MORPH_CLOSE, morph_kernel)

            # 3. NEW: split the depth-band mask into spatially-disconnected
            #    sub-regions. A blob that visually fused two people's bodies
            #    can still have two separate head-height clusters -- this is
            #    what recovers them as two head detections instead of one.
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                band_mask, connectivity=8
            )
            surviving_sub_regions = [
                i for i in range(1, num_labels)
                if stats[i, cv2.CC_STAT_AREA] >= self.min_head_area
            ]
            blob_was_split = len(surviving_sub_regions) > 1

            for i in surviving_sub_regions:
                sub_mask = (labels == i)

                head_mask[sub_mask] = 255

                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                w = int(stats[i, cv2.CC_STAT_WIDTH])
                h = int(stats[i, cv2.CC_STAT_HEIGHT])
                area = int(stats[i, cv2.CC_STAT_AREA])

                ys, xs = np.where(sub_mask)
                centroid = (int(xs.mean()), int(ys.mean()))

                # Highest point (smallest y) of this head sub-region -- the
                # "top head point" the paper projects onto the ground plane
                # in Section 4[cite: 1].
                top_idx = int(np.argmin(ys))
                top_point = (int(xs[top_idx]), int(ys[top_idx]))

                heads.append({
                    'head_id': head_counter,
                    'blob_id': blob['id'],
                    'split_from_shared_blob': blob_was_split,
                    'mask': sub_mask,
                    'bbox': (x, y, w, h),
                    'centroid': centroid,
                    'top_point': top_point,
                    'depth_min': d_hat,
                    'area': area,
                })
                head_counter += 1

        return head_mask, heads