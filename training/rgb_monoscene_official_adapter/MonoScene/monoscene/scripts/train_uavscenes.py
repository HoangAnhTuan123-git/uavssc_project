import os
import json

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from monoscene.data.uavscenes.params import (
    uavscenes_class_names,
    uavscenes_default_class_frequencies,
)
from monoscene.data.uavscenes.uav_dm import UAVScenesDataModule
from monoscene.models.monoscene import MonoScene

hydra.output_subdir = None


def _compute_class_frequencies(npz_paths, n_classes, cache_path=None):
    counts = np.zeros(n_classes, dtype=np.int64)
    for path in npz_paths:
        arr = np.load(path, allow_pickle=False)
        target = arr["target"]
        valid = target[target != 255]
        if valid.size == 0:
            continue
        binc = np.bincount(valid.reshape(-1), minlength=n_classes)
        counts += binc[:n_classes]
    counts = counts.astype(np.float64)
    counts[counts <= 0] = 1.0
    if cache_path is not None:
        np.save(cache_path, counts)
    return counts


def _partial_load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model_state = model.state_dict()

    loadable = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            loadable[key] = value
        else:
            skipped.append(key)

    model_state.update(loadable)
    model.load_state_dict(model_state)

    print("Loaded {} tensors from checkpoint".format(len(loadable)))
    print("Skipped {} tensors due to missing key or shape mismatch".format(len(skipped)))
    for k in skipped[:20]:
        print("  skipped:", k)
    if len(skipped) > 20:
        print("  ...")


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
    scene_filter = _maybe_parse_scene_filter(config.scene_filter)
    split_files = getattr(config, "split_files", None)

    batch_size = int(config.batch_size)
    n_gpus = int(config.n_gpus)
    num_workers = int(config.num_workers_per_gpu)

    data_module = UAVScenesDataModule(
        preprocess_root=config.uav_preprocess_root,
        project_scale=int(config.project_scale),
        frustum_size=int(config.frustum_size),
        batch_size=batch_size,
        num_workers=num_workers,
        split_ratios=(float(config.train_ratio), float(config.val_ratio), float(config.test_ratio)),
        scene_filter=scene_filter,
        data_root=getattr(config, "uav_data_root", None),
        split_files=split_files,
    )
    data_module.setup()

    train_paths = data_module.train_ds.get_sample_paths()
    freq_cache_path = os.path.join(config.uav_preprocess_root, "uav_class_frequencies.npy")
    if os.path.isfile(freq_cache_path):
        class_freq = np.load(freq_cache_path)
    else:
        class_freq = _compute_class_frequencies(
            train_paths,
            n_classes=len(uavscenes_class_names),
            cache_path=freq_cache_path,
        )

    class_weights = torch.from_numpy(1.0 / np.log(class_freq + 1.001))

    full_scene_size = (
        int(config.full_scene_size[0]),
        int(config.full_scene_size[1]),
        int(config.full_scene_size[2]),
    )

    project_res = ["1"]
    if bool(config.project_1_2):
        project_res.append("2")
    if bool(config.project_1_4):
        project_res.append("4")
    if bool(config.project_1_8):
        project_res.append("8")

    model = MonoScene(
        dataset="kitti",
        frustum_size=int(config.frustum_size),
        project_scale=int(config.project_scale),
        n_relations=int(config.n_relations),
        fp_loss=bool(config.fp_loss),
        feature=int(config.feature),
        full_scene_size=full_scene_size,
        project_res=project_res,
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
    )

    if bool(config.load_pretrained) and os.path.isfile(config.pretrained_model_path):
        _partial_load_checkpoint(model, config.pretrained_model_path)
    elif bool(config.load_pretrained):
        raise FileNotFoundError(
            "pretrained_model_path does not exist: {}".format(config.pretrained_model_path)
        )

    exp_name = config.exp_prefix
    exp_name += "_uavscenes_{}".format(config.run)
    exp_name += "_scene{}".format("all" if scene_filter is None else "-".join(scene_filter))
    exp_name += "_fs{}_{}_{}".format(*full_scene_size)
    exp_name += "_bs{}_lr{}".format(batch_size, config.lr)
    if config.context_prior:
        exp_name += "_3DCRP"
    if config.relation_loss:
        exp_name += "_CERel"
    if config.fp_loss:
        exp_name += "_fpLoss"

    logger = TensorBoardLogger(save_dir=config.uav_logdir, name=exp_name, version="")
    checkpoint_callbacks = [
        ModelCheckpoint(
            save_last=True,
            monitor="val/mIoU",
            save_top_k=1,
            mode="max",
            filename="{epoch:03d}-{val/mIoU:.5f}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer_kwargs = dict(
        callbacks=checkpoint_callbacks,
        deterministic=False,
        max_epochs=int(config.max_epochs),
        logger=logger,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        weights_summary="top",
    )

    if n_gpus <= 1:
        trainer_kwargs["gpus"] = 1 if torch.cuda.is_available() else 0
    else:
        trainer_kwargs["gpus"] = n_gpus
        trainer_kwargs["accelerator"] = "ddp"
        trainer_kwargs["sync_batchnorm"] = True

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
