import torch
import numpy as np
from mayavi import mlab
from monoscene.models.monoscene import MonoScene

import cv2

print("1. Loading the 1.5M Parameter Brain...")
# Point this to your local checkpoint!
ckpt_path = r"C:\Study\Uni\B3\INTERN\MonoScenePre\MonoScene\outputs\2026-04-21\19-18-14\uav_logs\uav_monoscene_uavscenes_1_sceneAMtown01_fs128_128_32_bs1_lr0.0001_3DCRP\checkpoints\epoch=000-val\mIoU=0.02667.ckpt"

# We load on CPU so your laptop's GPU doesn't crash from memory overload!
model = MonoScene.load_from_checkpoint(ckpt_path, map_location=torch.device('cpu'), strict=False)
model.eval()

print("2. Faking the Drone Eyes (Bypassing the DataLoader)...")
# 1. Load your real image
img_path = r"C:\Study\Uni\B3\INTERN\MonoScenePre\UAVScenes\AMtown01\interval5_CAM\1658137057.641204937.jpg"
raw_img = cv2.imread(img_path)

if raw_img is None:
    print(f"Error: Could not load image at {img_path}. Check the path!")
else:
    # 2. Resize to (Width: 448, Height: 256) and convert BGR to RGB
    raw_img = cv2.resize(raw_img, (448, 256))
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    # 3. Normalize and convert to Tensor
    # Scale to [0, 1] then apply standard ImageNet normalization
    img_np = raw_img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std
    
    # Change shape from (H, W, C) to (C, H, W) and add Batch dimension
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()

    batch = {
        'img': img_tensor, 
        'projected_pix_2': torch.zeros(1, 64*64*16, 2).long(), 
        'fov_mask_2': torch.ones(1, 64*64*16).bool(), 
    }

# We construct the exact mathematical tensors the PyTorch forward pass requires
#batch = {
#    'img': torch.zeros(1, 3, 256, 448), # A fake, completely black 2D image
#    'projected_pix_2': torch.zeros(1, 64*64*16, 2).long(), # Fake ray-tracing coordinates
#    'fov_mask_2': torch.ones(1, 64*64*16).bool(), # Fake Field of View mask
#}

print("3. Running the PyTorch Forward Pass (This might take a few seconds on CPU)...")
with torch.no_grad():
    output = model(batch)
    
    # Let's peek inside the box to see what the authors actually named it!
    if isinstance(output, dict):
        print(f"Success! The model returned these keys: {output.keys()}")
        # Smart extraction to catch singular vs plural naming
        if 'ssc_logits' in output:
            logits = output['ssc_logits']
        elif 'ssc_logit' in output:
            logits = output['ssc_logit']
        else:
            # Fallback: grab the first available tensor
            logits = list(output.values())[0] 
    else:
        logits = output
        
    # Run Argmax to pick the highest probability for each voxel
    y_pred = torch.argmax(logits, dim=1)

print("4. Extracting the 3D Voxel Grid...")
# Convert PyTorch tensor back to a normal Numpy grid
prediction_grid = y_pred[0].numpy().astype(np.uint8)

print("5. Rendering in Mayavi...")
# Filter out Class 0 (Empty Space) and Class 255 (Unknown)
mask = (prediction_grid > 0) & (prediction_grid < 255)
x, y, z = np.where(mask)
class_ids = prediction_grid[mask]

print(f"\n--- RESULTS ---")
print(f"The Network predicted {len(x)} solid voxels.")

# --- THE VTK BUG FIX ---
# Convert from integers to 32-bit floats so the color engine doesn't crash
x = x.astype(np.float32)
y = y.astype(np.float32)
z = z.astype(np.float32)
class_ids = class_ids.astype(np.float32)
# -----------------------

# Draw the 3D map
mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))
if len(x) > 0:
    # Locked vmin=1 and vmax=17 so the colors map to the exact UAVScenes classes
    mlab.points3d(x, y, z, class_ids, colormap="jet", mode="cube", scale_factor=1, vmin=1, vmax=17)
else:
    print("CONFIRMED: The AI predicted 100% Empty Space! Opening blank canvas...")
    
mlab.show()