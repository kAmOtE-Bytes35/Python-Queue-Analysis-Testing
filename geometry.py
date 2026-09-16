import numpy as np
from parameters import K_MAT, BASELINE_PLANE

class CameraGeometry:
    def __init__(self):
        self.fx = K_MAT[0, 0]
        self.fy = K_MAT[1, 1]
        self.cx = K_MAT[0, 2]
        self.cy = K_MAT[1, 2]
        self.baseline_plane = BASELINE_PLANE

    def backproject_pixel(self, u: int, v: int, depth: float) -> np.ndarray:
        """Converts a single (u, v) pixel and its depth to a 3D (X, Y, Z) point."""
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        return np.array([x, y, depth])

    def backproject_point_cloud(self, depth_image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Converts masked pixels in a depth image to a Nx3 array of 3D points."""
        if mask is None:
            mask = depth_image > 0
            
        v, u = np.where(mask)
        z = depth_image[v, u].astype(np.float32)
        
        valid = z > 0
        u, v, z = u[valid], v[valid], z[valid]
        
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        return np.column_stack((x, y, z))

    @staticmethod
    def distance_to_plane(point_3d: np.ndarray, plane: np.ndarray) -> float:
        """Calculates the orthogonal distance from a 3D point to a plane [a, b, c, d]."""
        a, b, c, d = plane
        x, y, z = point_3d
        numerator = abs(a*x + b*y + c*z + d)
        denominator = np.sqrt(a**2 + b**2 + c**2)
        return numerator / denominator


class RansacPlaneFitter:
    def __init__(self, distance_threshold: float = 30.0, max_iterations: int = 1000):
        self.distance_threshold = distance_threshold
        self.max_iterations = max_iterations

    def fit(self, points: np.ndarray):
        """Fits a plane aX + bY + cZ + d = 0 using RANSAC."""
        # Subsample points for fast iteration during RANSAC candidate search
        if len(points) > 10000:
            idx = np.random.choice(len(points), 10000, replace=False)
            process_points = points[idx]
        else:
            process_points = points

        if len(process_points) < 3:
            return None, None

        best_plane = None
        best_inlier_count = 0
        best_inliers = None
        num_points = len(process_points)

        for _ in range(self.max_iterations):
            idx = np.random.choice(num_points, 3, replace=False)
            p1, p2, p3 = process_points[idx]

            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            
            norm_length = np.linalg.norm(normal)
            if norm_length < 1e-6:
                continue 
                
            normal = normal / norm_length
            a, b, c = normal
            d = -np.dot(normal, p1)
            
            distances = np.abs(np.dot(process_points, normal) + d)
            inliers = distances < self.distance_threshold
            inlier_count = np.sum(inliers)
            
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_plane = np.array([a, b, c, d])
                best_inliers = inliers

        # Refine fitted plane using Least Squares on inliers
        if best_plane is not None and best_inlier_count >= 3:
            inlier_points = process_points[best_inliers]
            centroid = np.mean(inlier_points, axis=0)
            centered = inlier_points - centroid
            
            cov_matrix = np.dot(centered.T, centered)
            _, _, vt = np.linalg.svd(cov_matrix)
            
            normal = vt[2, :]
            d = -np.dot(normal, centroid)
            
            if normal[2] > 0:
                normal = -normal
                d = -d
                
            best_plane = np.array([normal[0], normal[1], normal[2], d])

        # Calculate inliers for ALL input points using the final refined plane
        if best_plane is not None:
            all_distances = np.abs(np.dot(points, best_plane[:3]) + best_plane[3])
            all_inliers = all_distances < self.distance_threshold
            return best_plane, all_inliers

        return None, None