# Hardware planning for UAVSSC training

## Conservative starting point
- 1 × 24 GB GPU
- 8 CPU cores
- 64 GB RAM
- 1 TB NVMe SSD

Good for:
- MonoScene fine-tuning
- LMSCNet-style LiDAR baseline
- SCPNet-style dense starter
- VoxFormer-style starter with batch size 1

## Comfortable setup
- 1 × 48 GB GPU
- 16 CPU cores
- 96–128 GB RAM
- 2 TB NVMe SSD

Good for:
- CGFormer-style RGB baseline
- RGB+LiDAR fusion
- larger image sizes or more aggressive ablations

## Safest / fastest setup
- 1 × 80 GB GPU (or 2 × 48 GB GPUs)
- 16+ CPU cores
- 128 GB RAM
- 2 TB NVMe SSD

Good for:
- full ablations
- larger crops / higher feature widths
- closer-to-official aerial SSC experiments
