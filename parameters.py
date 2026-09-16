import numpy as np

# Camera Intrinsic Matrix (K) converted from parameters.m
K_MAT = np.array([
    [529.21508098293293, 0.0, 328.94272028759258],
    [0.0, 525.56393630057437, 267.48068171871557],
    [0.0, 0.0, 1.0]
])

# Baseline ground plane [a, b, c, d] converted from parameters.m[cite: 8]
BASELINE_PLANE = np.array([0.05185302, -1.6952257, -1.0, 5681.561])