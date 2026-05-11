import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

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


def _project_res_from_config(config):
    project_res = ["1"]
    if bool(config.project_1_2):
        project_res.append("2")
    if bool(config.project_1_4):
        project_res.append("4")
    if bool(config.project_1_8):
        project_res.append("8")
    return project_res


def _build_model(config):
    full_scene_size = (
        int(config.full_scene_size[0]),
        int(config.full_scene_size[1]),
        int(config.full_scene_size[2]),
    )
    class_weights = _load_class_weights(config.uav_preprocess_root, len(uavscenes_class_names))
    model = MonoScene(
        dataset="kitti",
        feature=int(config.feature),
        project_scale=int(config.project_scale),
        fp_loss=bool(config.fp_loss),
        full_scene_size=full_scene_size,
        project_res=_project_res_from_config(config),
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
    return model

def _select_loader(data_module, split):
    split = str(split).lower()
    if split == "train":
        return data_module.train_dataloader()
    if split == "test":
        return data_module.test_dataloader()
    return data_module.val_dataloader()


def _move_batch_for_forward(batch, device):
    # MonoScene.forward only needs img and projection tensors. Projection tensors are
    # moved inside MonoScene.forward to the feature device, but img must be on device.
    batch = dict(batch)
    batch["img"] = batch["img"].to(device, non_blocking=True)
    return batch


@hydra.main(config_name="../config/uavscenes.yaml")
def main(config: DictConfig):
    torch.set_grad_enabled(False)
    ckpt = str(config.eval_checkpoint_path)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError("eval_checkpoint_path does not exist: {}".format(ckpt))

    out_root = Path(str(getattr(config, "predict_output_root", "predictions/uavscenes")))
    out_root.mkdir(parents=True, exist_ok=True)

    scene_filter = _maybe_parse_scene_filter(config.scene_filter)
    split_files = getattr(config, "split_files", None)
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
    loader = _select_loader(data_module, getattr(config, "eval_split", "val"))

    device = torch.device("cuda" if torch.cuda.is_available() and int(config.n_gpus) > 0 else "cpu")
    model = _build_model(config).to(device).eval()

    max_samples = int(getattr(config, "max_predict_samples", 50))
    use_amp = device.type == "cuda" and int(getattr(config, "precision", 32)) == 16

    wrote = 0
    for batch_idx, batch in enumerate(loader):
        if max_samples > 0 and wrote >= max_samples:
            break
        fwd_batch = _move_batch_for_forward(batch, device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(fwd_batch)
            logits = out["ssc_logit"]
            prob = torch.softmax(logits, dim=1)
            conf, pred = torch.max(prob, dim=1)

        pred_np = pred[0].detach().cpu().numpy().astype(np.uint8)
        conf_np = conf[0].detach().cpu().numpy().astype(np.float16)
        target_np = batch["target"][0].detach().cpu().numpy().astype(np.uint8)

        seq = str(batch["sequence"][0])
        frame_id = str(batch["frame_id"][0])
        out_dir = out_root / seq
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{frame_id}.npz"
        np.savez_compressed(
            out_path,
            pred=pred_np,
            confidence=conf_np,
            target=target_np,
            sequence=np.asarray(seq),
            frame_id=np.asarray(frame_id),
            img_path=np.asarray(str(batch.get("img_path", [""])[0])),
            image_shape_hw=batch.get("image_shape_hw", [torch.tensor([0, 0])])[0].cpu().numpy().astype(np.int32),
            projection_shape_hw=batch.get("projection_shape_hw", [torch.tensor([0, 0])])[0].cpu().numpy().astype(np.int32),
            rgb_backbone=np.asarray(str(getattr(config, "rgb_backbone", ""))),
            input_image_hw=np.asarray(list(getattr(config, "input_image_hw", [0, 0])), dtype=np.int32),
        )
        wrote += 1
        if wrote % 10 == 0 or wrote == 1:
            print(f"Wrote {wrote} predictions; latest: {out_path}")

    print(f"Done. Wrote {wrote} prediction files to {out_root}")


if __name__ == "__main__":
    main()
