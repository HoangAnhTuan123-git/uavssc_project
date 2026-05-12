import numpy as np
import open3d as o3d

# 1. Target your specific file
npz_file = "1658137073.640511673.npz"

print(f"Loading {npz_file}...")
data = np.load(npz_file)
print("Keys inside this file:", data.files)

voxel_grid = data['pred'] if 'pred' in data.files else data['y_pred']

# Find non-empty voxels
x, y, z = np.where(voxel_grid > 0)

# 2. Color Map (normalized 0.0 to 1.0)
colors_dict = {
    1: [1.0, 0.0, 0.0],       # roof -> Red
    2: [0.55, 0.27, 0.07],    # dirt road -> Brown
    3: [0.2, 0.2, 0.2],       # paved road -> Dark Grey
    4: [0.0, 0.0, 1.0],       # river -> Blue
    5: [0.68, 0.85, 0.9],     # pool -> Light Blue
    6: [1.0, 1.0, 0.0],       # bridge -> Yellow
    10: [0.56, 0.93, 0.56],   # green field -> Light Green
    11: [0.5, 0.5, 0.0],      # wild field -> Olive Green
    16: [1.0, 0.0, 1.0],      # paved walk -> Magenta
}

# Map the colors
colors = np.array([colors_dict.get(voxel_grid[cx, cy, cz], [0.8, 0.8, 0.8]) for cx, cy, cz in zip(x, y, z)])

print("Building solid Minecraft blocks...")
Z_STRETCH = 3.0  # Makes the buildings look tall without causing gaps!

# 3. Define the 8 corners of a single standard block
base_corners = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
], dtype=np.float64)

# Stretch the block itself, not just the spacing
base_corners[:, 2] *= Z_STRETCH

# Define the 12 triangles that make up the walls of the block
base_triangles = np.array([
    [0, 2, 1], [1, 2, 3], # bottom
    [4, 5, 6], [5, 7, 6], # top
    [0, 1, 4], [1, 5, 4], # front
    [2, 6, 3], [3, 6, 7], # back
    [0, 4, 2], [2, 4, 6], # left
    [1, 3, 5], [3, 7, 5]  # right
], dtype=np.int32)

# 4. Fast Numpy Vectorization to build the whole map instantly
N = len(x)
offsets = np.column_stack((x, y, z * Z_STRETCH))

# Build all vertices
all_vertices = offsets[:, np.newaxis, :] + base_corners[np.newaxis, :, :]
vertices = all_vertices.reshape(-1, 3)

# Build all triangles
vertex_offsets = np.arange(N) * 8
all_triangles = base_triangles[np.newaxis, :, :] + vertex_offsets[:, np.newaxis, np.newaxis]
triangles = all_triangles.reshape(-1, 3)

# Apply colors to all vertices
all_colors = np.repeat(colors, 8, axis=0)

# 5. Create the Open3D Mesh
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(triangles)
mesh.vertex_colors = o3d.utility.Vector3dVector(all_colors)

# Compute shadows and lighting
mesh.compute_vertex_normals()

# 6. Rotate the map so it stands up correctly! (Open3D uses Y as UP, our data uses Z as UP)
R = mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
mesh.rotate(R, center=(0, 0, 0))

print("Launching viewer! Try zooming in now.")
o3d.visualization.draw_geometries([mesh], window_name="True Solid Voxel Viewer", mesh_show_back_face=True)