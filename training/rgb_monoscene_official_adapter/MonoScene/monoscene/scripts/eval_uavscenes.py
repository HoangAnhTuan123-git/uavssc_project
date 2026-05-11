import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from monoscene.data.uavscenes.params import (
    uavscenes_class_names,
    uavscenes_default_class_frequencies,
)
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


def _load_class_weights(preprocess_root, n_classes):
    freq_cache_path = os.path.join(preprocess_root, "uav_class_frequencies.npy")
    if os.path.isfile(freq_cache_path):
        freq = np.load(freq_cache_path).astype(np.float64)
    else:
        freq = uavscenes_default_class_frequencies.astype(np.float64)
    if freq.shape[0] < n_classes:
        freq = np.pad(freq, (0, n_classes - freq.shape[0]), constant_values=1.0)
    freq = freq[:n_classes]
    freq[freq <= 0] = 1.0
    return torch.from_numpy(1.0 / np.log(freq + 1.001)).float()


@hydra.main(config_name="../config/uavscenes.yaml")
def main(config: DictConfig):
    torch.set_grad_enabled(False)

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
        input_image_hw=getattr(config, "input_image_hw", None),
    )
    data_module.setup()

    project_res = ["1"]
    if bool(config.project_1_2):
        project_res.append("2")
    if bool(config.project_1_4):
        project_res.append("4")
    if bool(config.project_1_8):
        project_res.append("8")

    class_weights = _load_class_weights(config.uav_preprocess_root, len(uavscenes_class_names))

    # Build the model manually instead of MonoScene.load_from_checkpoint.
    # This avoids PyTorch 2.6 weights_only checkpoint-loading issues in older Lightning.
    model = MonoScene(
        dataset="kitti",
        feature=int(config.feature),
        project_scale=int(config.project_scale),
        fp_loss=bool(config.fp_loss),
        full_scene_size=full_scene_size,
        project_res=project_res,
        frustum_size=int(config.frustum_size),
        n_relations=int(config.n_relations),
        n_classes=len(uavscenes_class_names),
        class_names=uavscenes_class_names,
        context_prior=bool(config.context_prior),
        relation_loss=bool(config.relation_loss),
        CE_ssc_loss=bool(config.CE_ssc_loss),
        sem_scal_loss=bool(config.sem_scal_loss),
        geo_scal_loss=bool(config.geo_scal_loss),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
        class_weights=class_weights,
        rgb_backbone=getattr(config, "rgb_backbone", "tf_efficientnet_b0_ns"),
        rgb_pretrained=bool(getattr(config, "rgb_pretrained", False)),
        freeze_rgb_encoder=bool(getattr(config, "freeze_rgb_encoder", False)),
    )
    ckpt = torch.load(config.eval_checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    trainer_kwargs = dict(deterministic=True)
    if hasattr(config, "precision"):
        trainer_kwargs["precision"] = int(config.precision)

    n_gpus = int(config.n_gpus)
    if n_gpus <= 1:
        trainer_kwargs["gpus"] = 1 if torch.cuda.is_available() else 0
    else:
        trainer_kwargs["gpus"] = n_gpus
        trainer_kwargs["accelerator"] = "ddp"
        trainer_kwargs["sync_batchnorm"] = True

    trainer = Trainer(**trainer_kwargs)
    split = str(getattr(config, "eval_split", "val")).lower()
    if split == "test":
        loader = data_module.test_dataloader()
    elif split == "train":
        loader = data_module.train_dataloader()
    else:
        loader = data_module.val_dataloader()
    trainer.test(model, test_dataloaders=loader)


if __name__ == "__main__":
    main()
