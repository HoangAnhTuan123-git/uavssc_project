# Training protocols

Recommended sequence:
1. RGB-only MonoScene baseline
2. LiDAR-only LMSCNet-style baseline
3. LiDAR-only SCPNet-style baseline
4. RGB-only CGFormer-style baseline
5. RGB-only VoxFormer-style baseline
6. RGB+LiDAR fusion baseline

Main reporting protocol:
- strict cross-scene folds

Auxiliary protocol:
- same-scene unseen-run

Always record:
- config file
- split file
- checkpoint path
- seed
- number of epochs
- hardware
