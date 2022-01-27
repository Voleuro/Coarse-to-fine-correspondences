import torch
from model.KPConv.architectures import architectures
from model.Models.roughmatching import RoughMatchingModel
from easydict import EasyDict as edict
from configs.config_utils import load_config
import numpy as np
from dataset.common import to_tsfm, get_correspondences, to_o3d_pcd
import os
from model.KPConv.preprocessing import collate_fn_descriptor, calibrate_neighbors
from lib.utils import get_fine_grained_correspondences
import time
import argparse


def point_cloud_from_ply(path):
    """
        Loads a point cloud from a ply

    :param path: Input ply path
    :return: Point cloud as (N,3) numpy array
    """

    return np.ones((150, 3))


def convert_point_clouds_to_dataset(config, src_path, target_path, points_lim=30000):
    """
        Adapted from the method __getitem__ in tdmatch.py
    :return:
    """

    print("Loading point clouds..")

    # As the rotations are required: Initialize them as identity (FIXME: Is this a valid assumption?)
    rot = np.identity(3)
    trans = np.array([0,0,0]).reshape((3,1))

    # Both point clouds need to be of shape (N,3)
    #src_pcd = point_cloud_from_ply(src_path)
    #tgt_pcd = point_cloud_from_ply(target_path)
    src_pcd = torch.load(src_path)
    tgt_pcd = torch.load(target_path)

    # if we get too many points, we do some downsampling
    if (src_pcd.shape[0] > points_lim):
        idx = np.random.permutation(src_pcd.shape[0])[:points_lim]
        src_pcd = src_pcd[idx]

    if (tgt_pcd.shape[0] > points_lim):
        idx = np.random.permutation(tgt_pcd.shape[0])[:points_lim]
        tgt_pcd = tgt_pcd[idx]

    if (trans.ndim == 1):
        trans = trans[:, None]

    # get correspondences
    correspondences = None
    src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
    tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
    rot = rot.astype(np.float32)
    trans = trans.astype(np.float32)

    return src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, correspondences, src_pcd, tgt_pcd, torch.ones(1)


def estimate_correspondences(inputs, config):
    config.model.eval()
    total_corr_num = 0

    with torch.no_grad():
        ##############################
        # Load inputs to device
        for k, v in inputs.items():
            if v is None:
                pass
            elif type(v) == list:
                inputs[k] = [items.to(config.device) for items in v]
            else:
                inputs[k] = v.to(config.device)

        #############################
        # Forward pass
        len_src_pcd = inputs['stack_lengths'][0][0]
        pcds = inputs['points'][0]
        src_pcd, tgt_pcd = pcds[:len_src_pcd], pcds[len_src_pcd:]

        len_src_nodes = inputs['stack_lengths'][-1][0]
        nodes = inputs['points'][-1]
        src_node, tgt_node = nodes[:len_src_nodes], nodes[len_src_nodes:]

        rot = inputs['rot']
        trans = inputs['trans']

        src_candidates_c, tgt_candidates_c, local_scores, node_corr, node_corr_conf, src_pcd_sel, tgt_pcd_sel = config.model.forward(inputs)

        total_corr_num += node_corr.shape[0]

        correspondences, corr_conf = get_fine_grained_correspondences(local_scores, mutual=False, supp=False, node_corr_conf=node_corr_conf)

        data = dict()
        data['src_pcd'], data['tgt_pcd'] = src_pcd.cpu(), tgt_pcd.cpu()
        data['src_node'], data['tgt_node'] = src_node.cpu(), tgt_node.cpu()
        data['src_candidate'], data['tgt_candidate'] = src_candidates_c.view(-1, 3).cpu(), tgt_candidates_c.view(-1, 3).cpu()
        data['src_candidate_id'], data['tgt_candidate_id'] = src_pcd_sel.cpu(), tgt_pcd_sel.cpu()
        data['rot'] = rot.cpu()
        data['trans'] = trans.cpu()
        data['correspondences'] = correspondences.cpu()
        data['confidence'] = corr_conf.cpu()

        return data


if __name__ == '__main__':
    print("Running 'CoFiNet: Reliable Coarse-to-fine Correspondences for Robust Point Cloud Registration'")
    start = time.time()

    # Parse the inputs
    parser = argparse.ArgumentParser(description="A commandline interface for 'CoFiNet: Reliable Coarse-to-fine Correspondences for Robust Point Cloud Registration'")
    parser.add_argument("source", type=str, help="Path to the source point cloud")
    parser.add_argument("target", type=str, help="Path to the target point cloud")
    parser.add_argument("output", type=str, help="Path to the output txt file containing the correspondences")
    args = parser.parse_args()

    # Init the config struct
    config = edict(load_config("configs/tdmatch/tdmatch_test.yaml"))

    # Enable / Disable the GPU
    config.gpu_mode = True

    # Dataset settings
    #config.voxel_size = 0.025
    #config.augment_noise = 0.005
    #config.pos_margin = 0.1
    #config.overlap_radius = 0.0375

    # Enable the GPU, if wished
    if config.gpu_mode:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')

    # Init the model
    print("Initializing Model..")
    config.architecture = architectures["tdmatch_full"]
    config.model = RoughMatchingModel(config)

    # Move the model onto the device
    config.model = config.model.to(config.device)

    # Read in the dataset
    data_tuple = convert_point_clouds_to_dataset(config, args.source, args.target)

    # Preprocess that dataset
    neighborhood_limits = calibrate_neighbors([data_tuple], config, collate_fn=collate_fn_descriptor)
    dataset_input_dict = collate_fn_descriptor([data_tuple], config, neighborhood_limits)

    # Execute the network
    print("Executing deep model..")
    result = estimate_correspondences(dataset_input_dict, config)

    # Write out the correspondences
    with open(args.output, "w") as file:
        for corr in result["correspondences"]:
            file.write(str(int(corr[0])) + ";" + str(int(corr[1])) + "\n")

    end = time.time()
    print("Calculated", len(result["correspondences"]), "correspondences (out of", max(len(data_tuple[7]), len(data_tuple[8])) , "points) in", end - start, "s")



