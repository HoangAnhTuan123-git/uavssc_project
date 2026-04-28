import torch


def collate_fn(batch):
    data = {}
    imgs = []
    CP_mega_matrices = []
    targets = []
    frame_ids = []
    sequences = []
    cam_ks = []
    cam_Es = []
    frustums_masks = []
    frustums_class_dists = []

    scale_3ds = batch[0]["scale_3ds"]
    for scale_3d in scale_3ds:
        data["projected_pix_{}".format(scale_3d)] = []
        data["fov_mask_{}".format(scale_3d)] = []
        data["pix_z_{}".format(scale_3d)] = []

    for input_dict in batch:
        cam_ks.append(torch.from_numpy(input_dict["cam_k"]).double())
        cam_Es.append(torch.from_numpy(input_dict["cam_E"]).float())

        if input_dict.get("frustums_masks") is not None:
            frustums_masks.append(torch.from_numpy(input_dict["frustums_masks"]))
        else:
            frustums_masks.append(None)

        if input_dict.get("frustums_class_dists") is not None:
            frustums_class_dists.append(
                torch.from_numpy(input_dict["frustums_class_dists"]).float()
            )
        else:
            frustums_class_dists.append(None)

        for key in data:
            data[key].append(torch.from_numpy(input_dict[key]))

        imgs.append(input_dict["img"])
        frame_ids.append(input_dict["frame_id"])
        sequences.append(input_dict["sequence"])
        targets.append(torch.from_numpy(input_dict["target"]))
        CP_mega_matrices.append(torch.from_numpy(input_dict["CP_mega_matrix"]))

    ret_data = {
        "frame_id": frame_ids,
        "sequence": sequences,
        "frustums_class_dists": frustums_class_dists,
        "frustums_masks": frustums_masks,
        "cam_k": cam_ks,
        "cam_E": cam_Es,
        "img": torch.stack(imgs),
        "CP_mega_matrices": CP_mega_matrices,
        "target": torch.stack(targets),
    }

    for key in data:
        ret_data[key] = data[key]

    return ret_data
