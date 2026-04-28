import numpy as np
from mayavi import mlab

#file_path = r"C:\Study\Uni\B3\INTERN\MonoScenePre\uavssc_monoscene_prep\artifacts\monoscene_preprocess_grounded\AMtown01\1658137057.641204937.npz"
#file_path = r"C:\Study\Uni\B3\INTERN\MonoScenePre\uavssc_monoscene_prep\artifacts\monoscene_preprocess_grounded\AMtown01\1658137805.109770119.npz"
file_path = r"C:\Study\Uni\B3\INTERN\MonoScenePre\uavssc_rgb_only_monoscene_preprocess\artifacts\monoscene_preprocess_grounded\AMtown01\1658137057.641204937.npz"

print("Loading Perfect 3D Ground Truth...")
data = np.load(file_path, allow_pickle=True)
voxel_grid = None

# --- THE HUNTER ---
for key in data.files:
    val = data[key]
    if getattr(val, 'shape', None) in [(), (1,)]:
        unwrapped = val.item()
        if isinstance(unwrapped, dict):
            for d_key, d_val in unwrapped.items():
                if isinstance(d_val, np.ndarray) and d_val.ndim == 3:
                    voxel_grid = d_val
                    break
    elif isinstance(val, np.ndarray) and val.ndim == 3:
        voxel_grid = val
        break

# 2. Filter out Empty Space
mask = (voxel_grid > 0) & (voxel_grid < 255)
x, y, z = np.where(mask)
class_ids = voxel_grid[mask]

print(f"Drawing {len(x)} solid voxels.")
print(f"Unique classes found: {np.unique(class_ids)}")

# --- THE VTK BUG FIX ---
# We force all numbers to be Floats so VTK's color engine doesn't crash!
x = x.astype(np.float32)
y = y.astype(np.float32)
z = z.astype(np.float32)
class_ids = class_ids.astype(np.float32)
# -----------------------

# 3. Draw the 3D Cubes
mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))
# vmin=1 and vmax=20 forces the rainbow to spread properly
mlab.points3d(x, y, z, class_ids, colormap="jet", mode="cube", scale_factor=1, vmin=1, vmax=20)
mlab.show()