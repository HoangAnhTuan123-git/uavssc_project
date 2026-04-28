
cd /mnt/c/Study/Uni/B3/INTERN/MonoScenePre/MonoScene
conda activate monoscene
python monoscene/scripts/train_uavscenes.py


Line 7: (uavscenes.yaml)
uav_preprocess_root: "/mnt/c/Study/Uni/B3/INTERN/MonoScenePre/uavssc_monoscene_prep/artifacts/monoscene_preprocess_grounded"

Line 18:(uavscenes.yaml)
pretrained_model_path: "/mnt/c/Study/Uni/B3/INTERN/MonoScenePre/MonoScene/trained_models/monoscene_kitti.ckpt"

In uav_dataset.py ( in data/uavscenes/) ( If using linux)
        # --- ADD THESE TWO LINES TO FIX HYDRA ---
        img_path = img_path.replace("\\", "/")
        img_path = img_path.replace("../UAVScenes", "/mnt/c/Study/Uni/B3/INTERN/MonoScenePre/UAVScenes")
        # ----------------------------------------    

        if not os.path.exists(img_path):
            raise FileNotFoundError(
                "Image path inside npz does not exist: {} (from {})".format(
                    img_path, npz_path
                )
            )