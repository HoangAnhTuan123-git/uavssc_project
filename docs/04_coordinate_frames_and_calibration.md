# Coordinate frames and calibration

Document the conventions you actually use:
- world frame
- camera frame
- LiDAR frame
- local voxel grid frame

Record:
- where camera-to-LiDAR calibration comes from
- where camera-to-map calibration comes from
- how `T_world_cam` and `T_world_lidar` are defined
- what `vox_origin` means in each export format

Also save visual QC examples:
- image + projected LiDAR
- image + projected occupied voxels
- scene voxel slice + semantic labels
