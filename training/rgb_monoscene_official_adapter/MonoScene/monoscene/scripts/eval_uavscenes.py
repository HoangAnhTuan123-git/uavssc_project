import os

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from monoscene.data.uavscenes.params import uavscenes_class_names
from monoscene.data.uavscenes.uav_dm import UAVScenesDataModule
from monoscene.models.monoscene import MonoScene

hydra.output_subdir = None


def _maybe_parse_scene_filter(scene_filter):
    if scene_filter is None:
        return None
    if isinstance(scene_filter, str):
        if scene_filter.strip() == "":
            return None
        return [x.strip() for x in scene_filter.split(",") if x.strip()]
    return scene_filter


@hydra.main(config_name="../config/uavscenes.yaml")
def main(config: DictConfig):
    if not os.path.isfile(config.eval_checkpoint_path):
        raise FileNotFoundError(
            "eval_checkpoint_path does not exist: {}".format(config.eval_checkpoint_path)
        )

    scene_filter = _maybe_parse_scene_filter(config.scene_filter)
    split_files = getattr(config, "split_files", None)
    full_scene_size = (
        int(config.full_scene_size[0]),
        int(config.full_scene_size[1]),
        int(config.full_scene_size[2]),
    )

    data_module = UAVScenesDataModule(
        preprocess_root=config.uav_preprocess_root,
        project_scale=int(config.project_scale),
        frustum_size=int(config.frustum_size),
        batch_size=1,
        num_workers=int(config.num_workers_per_gpu),
        split_ratios=(float(config.train_ratio), float(config.val_ratio), float(config.test_ratio)),
        scene_filter=scene_filter,
        data_root=getattr(config, "uav_data_root", None),
        split_files=split_files,
    )
    data_module.setup()

    model = MonoScene.load_from_checkpoint(
        config.eval_checkpoint_path,
        dataset="kitti",
        feature=int(config.feature),
        project_scale=int(config.project_scale),
        fp_loss=bool(config.fp_loss),
        full_scene_size=full_scene_size,
        n_classes=len(uavscenes_class_names),
        class_names=uavscenes_class_names,
    )
    model.eval()

    trainer_kwargs = dict(deterministic=True)
    n_gpus = int(config.n_gpus)
    if n_gpus <= 1:
        trainer_kwargs["gpus"] = 1 if torch.cuda.is_available() else 0
    else:
        trainer_kwargs["gpus"] = n_gpus
        trainer_kwargs["accelerator"] = "ddp"
        trainer_kwargs["sync_batchnorm"] = True

    trainer = Trainer(**trainer_kwargs)
    trainer.test(model, test_dataloaders=data_module.val_dataloader())


if __name__ == "__main__":
    main()
