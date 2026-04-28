from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

from monoscene.data.uavscenes.uav_dataset import UAVScenesDataset
from monoscene.data.uavscenes.collate import collate_fn
from monoscene.data.utils.torch_util import worker_init_fn


class UAVScenesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        preprocess_root,
        project_scale=2,
        frustum_size=4,
        batch_size=1,
        num_workers=4,
        split_ratios=(0.7, 0.15, 0.15),
        scene_filter=None,
        data_root=None,
        split_files=None,
    ):
        super(UAVScenesDataModule, self).__init__()
        self.preprocess_root = preprocess_root
        self.project_scale = project_scale
        self.frustum_size = frustum_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratios = split_ratios
        self.scene_filter = scene_filter
        self.data_root = data_root
        self.split_files = split_files or {}

    def setup(self, stage=None):
        common = dict(
            preprocess_root=self.preprocess_root,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            split_ratios=self.split_ratios,
            scene_filter=self.scene_filter,
            data_root=self.data_root,
        )
        self.train_ds = UAVScenesDataset(
            split="train",
            fliplr=0.5,
            color_jitter=(0.4, 0.4, 0.4),
            sample_list_path=self.split_files.get("train"),
            **common,
        )
        self.val_ds = UAVScenesDataset(
            split="val",
            fliplr=0.0,
            color_jitter=None,
            sample_list_path=self.split_files.get("val"),
            **common,
        )
        self.test_ds = UAVScenesDataset(
            split="test",
            fliplr=0.0,
            color_jitter=None,
            sample_list_path=self.split_files.get("test"),
            **common,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )
