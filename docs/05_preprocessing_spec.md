# Preprocessing specification

Recommended order:
1. inspect dataset
2. build taxonomy
3. build manifest
4. projection debug
5. fuse global semantic cloud
6. optionally build scene voxel grid cache
7. export RGB-only NPZ
8. export LiDAR-only NPZ
9. export fusion NPZ
10. run alignment overlays

Critical checks:
- LiDAR point count matches label count
- projected voxels align with RGB image
- free space is not confused with unknown
- scene split indexing is frozen before training
