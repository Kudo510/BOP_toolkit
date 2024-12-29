# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates masks of object models in the ground-truth poses."""

import os
import numpy as np

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer
from bop_toolkit_lib import visibility
from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
import json
from PIL import Image


# PARAMETERS.
################################################################################
p = {
    # See dataset_params.py for options.
    "dataset": ["liangan"], 
    # [
    #             "qiuxiao",
    #             "neixinlun",
    #             "neixinlun2",
    #             "zhouchengquan",
    #             "hudiejian",
    #             "daoliuzhao",
    #             "banjinjia",
    #             "liangan",
    #             "diaohuanluoshuan",
    #             "yuanguan",
    #             "lianjiejian",
    #             "hudiebanjin",
    #             "banjinjianlong",
    #             "zhijiaobanjin",
    #             "jingjiagongjian",
    #             "jiaojieyuanguan",
    #             "ganqiuxiao",
    #             "fanguangzhao",
    #             "lungufanlan"
    #         ],
    # Dataset split. Options: 'train', 'val', 'test'.
    "dataset_split": "scene000000", # , "scene000002", "scene000003", "scene000004"],
    # Dataset split type. None = default. See dataset_params.py for options.
    "dataset_split_type": None,
    # Tolerance used in the visibility test [mm].
    "delta": 0.1,  # 5 for ITODD, 15 for the other datasets.
    # Type of the renderer.
    "renderer_type": "vispy",  # Options: 'vispy', 'cpp', 'python'.
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
    "model_type" : None
}
################################################################################




def get_bounds(mask):
    """
    Returns bounding box as [x, y, width, height] for a binary mask
    where (x,y) is top-left corner and width/height are the dimensions
    
    Args:
        mask: 2D numpy array with True/1 for object, False/0 for background
    Returns:
        [x, y, width, height] of the bounding box
    """
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    
    min_x = int(cols.argmax())
    min_y = int(rows.argmax())
    max_x = int(len(cols) - cols[::-1].argmax() - 1)
    max_y = int(len(rows) - rows[::-1].argmax() - 1)
    
    bbox = [
        int(min_x),
        int(min_y),
        int(max_x - min_x),
        int(max_y - min_y)
    ]
    
    return bbox

for i in range(len(p["dataset"])):

    # Load dataset parameters.
    dp_model = dataset_params.get_model_params(
        p["datasets_path"], p["dataset"][i], p["model_type"]
    )

    models_info = {}
    for obj_id in dp_model["obj_ids"]:
        misc.log("Processing model of object {}...".format(obj_id))

        model = inout.load_ply(dp_model["model_tpath"].format(obj_id=obj_id))

        # Calculate 3D bounding box.
        ref_pt = model["pts"].min(axis=0).flatten()
        size = (model["pts"].max(axis=0) - ref_pt).flatten()

        # Calculated diameter.
        diameter = misc.calc_pts_diameter(model["pts"])

        models_info[obj_id] = {
            "min_x": ref_pt[0],
            "min_y": ref_pt[1],
            "min_z": ref_pt[2],
            "size_x": size[0],
            "size_y": size[1],
            "size_z": size[2],
            "diameter": diameter,
        }

    # Save the calculated info about the object models.
    inout.save_json(dp_model["models_info_path"], models_info)

    # Load dataset parameters.
    dp_split = dataset_params.get_split_params(
        p["datasets_path"], p["dataset"][i], p["dataset_split"], p["dataset_split_type"]
    )

    model_type = None
    if p["dataset"][i] == "tless":
        model_type = "cad"
    dp_model = dataset_params.get_model_params(p["datasets_path"], p["dataset"][i], model_type)

    scene_ids = dataset_params.get_present_scene_ids(dp_split)
    for scene_id in scene_ids:
        # Load scene GT.
        scene_gt_path = dp_split["scene_gt_tpath"].format(scene_id=scene_id)
        scene_gt = inout.load_scene_gt(scene_gt_path)

        # Load scene camera.
        scene_camera_path = dp_split["scene_camera_tpath"].format(scene_id=scene_id)
        scene_camera = inout.load_scene_camera(scene_camera_path)

        # Create folders for the output masks (if they do not exist yet).
        mask_dir_path = os.path.dirname(
            dp_split["mask_tpath"].format(scene_id=scene_id, im_id=0, gt_id=0)
        )
        misc.ensure_dir(mask_dir_path)

        mask_visib_dir_path = os.path.dirname(
            dp_split["mask_visib_tpath"].format(scene_id=scene_id, im_id=0, gt_id=0)
        )
        misc.ensure_dir(mask_visib_dir_path)

        # Initialize a renderer.
        misc.log("Initializing renderer...")

        rgb_path = f"{p['datasets_path']}/{p['dataset'][i]}/{p['dataset_split']}/{scene_id:06d}/rgb/000000.png"
        height, width, _ = np.array(Image.open(rgb_path)).shape
        # width, height = dp_split["im_size"]
        assert height > 3

        ren = renderer.create_renderer(
            width, height, renderer_type=p["renderer_type"], mode="depth"
        )

        # Add object models.
        for obj_id in dp_model["obj_ids"]:
            ren.add_object(obj_id, dp_model["model_tpath"].format(obj_id=obj_id))

        im_ids = sorted(scene_gt.keys())

        scene_gt_info = {}
        scene_gt_info_path = dp_split["scene_gt_info_tpath"].format(scene_id=scene_id)
        for im_id in im_ids:
            if im_id % 100 == 0:
                misc.log(
                    "Calculating masks - dataset: {} ({}, {}), scene: {}, im: {}".format(
                        p["dataset"][i],
                        p["dataset_split"],
                        p["dataset_split_type"],
                        scene_id,
                        im_id,
                    )
                )

            K = scene_camera[im_id]["cam_K"]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            # Load depth image.
            depth_path = dp_split["depth_tpath"].format(scene_id=scene_id, im_id=im_id)
            depth_im = inout.load_depth(depth_path)
            depth_im *= scene_camera[im_id]["depth_scale"]  # to [mm]
            dist_im_gt = misc.depth_im_to_dist_im_fast(depth_im, K)

            # Create our own depth image
            dist_im = list()
            for gt_id, gt in enumerate(scene_gt[im_id]):
                depth_gt = ren.render_object(
                    gt["obj_id"], gt["cam_R_m2c"], gt["cam_t_m2c"], fx, fy, cx, cy
                )["depth"]
                dist_gt = misc.depth_im_to_dist_im_fast(depth_gt, K)
                modified_dist_gt = np.where(dist_gt==0, 100000, dist_gt)
                inout.save_im("xoa4.png", dist_gt.astype(np.uint8))
                dist_im.append(modified_dist_gt)
            dist_im = np.stack(np.array(dist_im))
            min_dist_im = np.min(dist_im, axis=0)
            # inout.save_im("xoa4.png", min_dist_im.astype(np.uint8))

            # max_dist_im = np.where(dist_gt==100000, 0, dist_gt)
            # for gt_id, dist_gt in enumerate(dist_im):
            scene_gt_info[im_id] = []
            for gt_id in range(len(dist_im)):
            # for gt_id in range(9, len(scene_gt[im_id])): #[0,1,2,3,4,5,6,7,8,11,10,9,12,13,14,15,16,17,18,19,20]:
                dist_gt = dist_im[gt_id]
                # dist_gt = np.where(dist_gt==0, 100000, dist_gt)
                # Mask of the full object silhouette.
                mask = dist_gt <100000

                # Mask of the visible part of the object silhouette.
                # mask_visib = np.logical_and(
                #     np.logical_or(dist_gt== min_dist_im, min_dist_im == 0), dist_gt > 0
                # )
                mask_visib = np.logical_and(dist_gt == min_dist_im, min_dist_im != 100000) # np.logical_and(dist_gt<min_dist_im, min_dist_im !=100000 )
                # inout.save_im("xoa4.png", 255 * mask_visib.astype(np.uint8))

                # Save the calculated masks.
                mask_path = dp_split["mask_tpath"].format(
                    scene_id=scene_id, im_id=im_id, gt_id=gt_id
                )
                inout.save_im(mask_path, 255 * mask.astype(np.uint8))

                mask_visib_path = dp_split["mask_visib_tpath"].format(
                    scene_id=scene_id, im_id=im_id, gt_id=gt_id
                )
                inout.save_im(mask_visib_path, 255 * mask_visib.astype(np.uint8))


                bbox_obj = get_bounds(mask)
                bbox_visib = get_bounds(mask_visib)

                px_count_all = np.sum(mask > 0) # just basically the pixels of the masks

                # Number of pixels in the object silhouette with a valid depth measurement
                # (i.e. with a non-zero value in the depth image).
                ## BUt since the depth map is incorrect- we use the K,I, it means the masks one will always be valid for all pixels
                px_count_valid = np.sum(mask > 0) # = np.sum(dist_im_gt[mask] > 0) # Assuming all projected pixels are valid - ja then it is the same- for other s.t icbin we have this abit smaller px_count_all - maybe thy apply some filtering/validation for the pixel - to remove several pixels- but actually if all are valid then it should be the same as px_count_all 
                px_count_visib = np.sum(mask_visib > 0)
                visib_fract = px_count_visib / px_count_all if px_count_all > 0 else 0

                scene_gt_info[im_id].append({
                    "bbox_obj": bbox_obj,
                    "bbox_visib": bbox_visib,
                    "px_count_all": int(px_count_all),
                    "px_count_valid": int(px_count_valid),
                    "px_count_visib": int(px_count_visib),
                    "visib_fract": float(visib_fract)
                })

            # Save scene_gt_info.json
        with open(scene_gt_info_path, 'w') as f:
            json.dump(scene_gt_info, f, indent=2)

