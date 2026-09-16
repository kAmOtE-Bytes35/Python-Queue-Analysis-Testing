import cv2

# Replace with the path to one frame in your depth directory
depth_path = "C:\\Users\\alter\\OneDrive\\Desktop\\DLSU\\AY 2025 - 2026\\Term 3 AY 2025 - 2026\\COMPVIS1\\Project\\micc_crowd_counting\\micc_crowd_counting\\Queue\\depth\\0.png"

# IMREAD_UNCHANGED is critical to avoid automatic conversion to 8-bit RGB
depth_frame = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

if depth_frame is None:
    print(f"Error: Could not read image at {depth_path}")
else:
    print(f"Data Type (dtype) : {depth_frame.dtype}")
    print(f"Shape             : {depth_frame.shape}")
    print(f"Min Depth Value   : {depth_frame.min()}")
    print(f"Max Depth Value   : {depth_frame.max()}")