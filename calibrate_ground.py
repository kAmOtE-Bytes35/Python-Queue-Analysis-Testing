import os
import cv2
import numpy as np
from geometry import CameraGeometry, RansacPlaneFitter

def calibrate_from_base_frame(dataset_path: str):
    # Path to the base frame (0.png)[cite: 4]
    base_frame_path = os.path.join(dataset_path, "depth", "0.png")
    
    print(f"Loading base frame: {base_frame_path}")
    depth_image = cv2.imread(base_frame_path, cv2.IMREAD_UNCHANGED)
    
    if depth_image is None:
        print("Error: Could not read base frame. Check the path.")
        return None

    geom = CameraGeometry()
    fitter = RansacPlaneFitter(distance_threshold=30.0, max_iterations=2000)

    print("Back-projecting depth pixels to 3D point cloud...")
    # Restrict mask to the lower 60% of the image to guarantee we hit the floor, not the walls
    h, w = depth_image.shape
    floor_roi_mask = np.zeros_like(depth_image, dtype=bool)
    floor_roi_mask[int(h*0.4):, :] = (depth_image[int(h*0.4):, :] > 0)
    
    point_cloud = geom.backproject_point_cloud(depth_image, mask=floor_roi_mask)
    
    print(f"Generated {len(point_cloud)} 3D points. Running RANSAC...")
    fitted_plane, inliers = fitter.fit(point_cloud)
    
    if fitted_plane is not None:
        inlier_ratio = np.sum(inliers) / len(point_cloud) * 100
        print("\n--- Calibration Complete ---")
        print(f"MATLAB Baseline Plane: {geom.baseline_plane}")
        print(f"RANSAC Fitted Plane:   {fitted_plane}")
        print(f"Inlier Ratio:          {inlier_ratio:.2f}%")
        return fitted_plane
    else:
        print("Failed to fit a plane.")
        return None

if __name__ == "__main__":
    # Ensure this points to your QUEUE dataset directory
    dataset_dir = "./QUEUE" 
    calibrate_from_base_frame(dataset_dir)