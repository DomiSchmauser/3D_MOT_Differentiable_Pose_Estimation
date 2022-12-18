import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import mathutils

from Tracking.utils.train_utils import convert_voxel_to_pc
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate


def box2minmax(corner_pt_box):
    '''
    Box from 8x3 to minmax format
    Only works properly for axis aligned boxes
    '''
    xyz_min = torch.min(corner_pt_box, dim=0).values
    xyz_max = torch.max(corner_pt_box, dim=0).values
    box = np.concatenate((xyz_min.numpy(), xyz_max.numpy()))
    return box

def box2minmax_axaligned(corner_pt_box):
    '''
    Box from 8x3 to minmax format
    For non-axis aligned boxes, first enclose with axis-aligned box, then calc minmax
    '''

    bbox3d_obj = o3d.geometry.AxisAlignedBoundingBox()
    bbox_3d = bbox3d_obj.create_from_points(o3d.utility.Vector3dVector(corner_pt_box))
    corner_pt_box = torch.from_numpy(np.array(bbox_3d.get_box_points()))
    xyz_min = torch.min(corner_pt_box, dim=0).values
    xyz_max = torch.max(corner_pt_box, dim=0).values
    box = np.concatenate((xyz_min.numpy(), xyz_max.numpy()))
    return box

def cad2world_mat(rot, loc, scale, with_scale=True):
    '''
    Return cad2world matrix from annotations
    '''
    cad2world = torch.eye(4)
    scale_mat = torch.diag(torch.tensor([scale, scale, scale]))
    if with_scale:
        cad2world[:3, :3] = scale_mat @ euler_to_rot(rot, fmt='torch')
    else:
        cad2world[:3, :3] = euler_to_rot(rot, fmt='torch')

    cad2world[:3, 3] = loc
    return cad2world

def euler_to_rot(euler_rot, fmt='torch', constraint=False):
    '''
    Euler to 3x3 Rotation Matrix transform
    '''

    if constraint:
        euler_rot = torch.tensor([0, 0, euler_rot[2]])
    euler = mathutils.Euler(euler_rot)
    rot = np.array(euler.to_matrix())

    if fmt == 'torch':
        return torch.from_numpy(rot)
    else:
        return rot


def visualize_graph(G, color):
    '''
    Visualise Graph data connectivity
    '''
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()



def fuse_pose(trajectories, seq_len=None):
    '''
    pose fusion via slurp and spline interpolation
    '''

    def get_scale(m):
        if type(m) == torch.Tensor:
            return m.norm(dim=0)
        return np.linalg.norm(m, axis=0)

    def fill_last(fill_list, exp_dim=False):
        # fill with last value
        for t_idx, tt in enumerate(fill_list):
            if tt is None:
                for i in range(t_idx - 1, -1, -1):
                    if fill_list[i] is not None:
                        if exp_dim:
                            fill_list[t_idx] = np.expand_dims(fill_list[i], axis=0)
                        else:
                            fill_list[t_idx] = fill_list[i]
                        break
        return fill_list

    def fill_last_t(fill_list, exp_dim=True):
        # fill with last value
        for t_idx, tt in enumerate(fill_list):
            if tt.sum() == 0:
                for i in range(t_idx - 1, -1, -1):
                    if fill_list[i].sum() != 0:
                        if exp_dim:
                            fill_list[t_idx] = np.expand_dims(np.squeeze(fill_list[i]), axis=0)
                        else:
                            fill_list[t_idx] = fill_list[i]
                        break
            else:
                fill_list[t_idx] = np.expand_dims(fill_list[t_idx], axis=0)

        for t_idx, tt in enumerate(fill_list):
            if len(tt.shape) == 1:
                fill_list[t_idx] = np.expand_dims(fill_list[t_idx], axis=0)

        return fill_list

    def unscale_mat(cad2world):

        c2w_cpy = torch.clone(cad2world)
        rot = cad2world[:3, :3]
        scale = get_scale(rot)
        unscaled_rot = rot / scale
        c2w_cpy[:3, :3] = unscaled_rot
        return c2w_cpy

    new_trajectories = []

    times = np.arange(seq_len)
    for traj in trajectories:
        key_times = []
        key_trans = []
        key_rots = []
        t_trans = [np.zeros(3) for i in range(seq_len)]
        t_vox = [None for i in range(seq_len)]
        t_box = [None for i in range(seq_len)]
        t_id = [None for i in range(seq_len)]
        t_scale = [None for i in range(seq_len)]
        for pred in traj:
            key_rots.append(torch.unsqueeze(unscale_mat(pred['obj']['cad2world'][:3, :3]), dim=0))
            key_trans.append(torch.unsqueeze(pred['obj']['cad2world'][:3, 3], dim=0))
            key_times.append(pred['scan_idx'])
            t_trans[pred['scan_idx']] = pred['obj']['cad2world'][:3, 3].numpy()
            t_vox[pred['scan_idx']] = pred['obj']['voxel']
            t_id[pred['scan_idx']] = pred['obj']['obj_idx']
            t_box[pred['scan_idx']] = pred['obj']['compl_box']
            t_scale[pred['scan_idx']] = get_scale(pred['obj']['cad2world'][:3, :3])

        times = np.linspace(key_times[0], key_times[-1], num=key_times[-1]-key_times[0]+1).astype(np.int)
        traj_rots = torch.cat(key_rots, dim=0).numpy()
        key_trans = torch.cat(key_trans, dim=0).numpy()

        t_trans = np.concatenate(fill_last_t(t_trans, exp_dim=True), axis=0)
        t_trans[:, 0] = gaussian_filter1d(t_trans[:, 0], 3)
        t_trans[:, 1] = gaussian_filter1d(t_trans[:, 1], 3)
        t_trans[:, 2] = gaussian_filter1d(t_trans[:, 2], 3)
        t_vox = fill_last(t_vox)
        t_id = fill_last(t_id)
        t_box = fill_last(t_box)
        t_scale = fill_last(t_scale)

        r = R.from_matrix(traj_rots)
        slerp = Slerp(key_times, r)
        interp_rots = slerp(times)
        interp_rotmat = interp_rots.as_matrix()
        euler_rots = interp_rots.as_euler('xyz')
        euler_rots[:,-1] = gaussian_filter1d(euler_rots[:,-1], 3) #3 = sigma = standard deviation
        euler_rots[:, -1] = np.clip(euler_rots[:,-1], euler_rots[0,-1] - (0.2 * euler_rots[0,-1]), euler_rots[0,-1] + (0.2 * euler_rots[0,-1]))
        r_e = R.from_euler('xyz', euler_rots, degrees=False)
        interp_rotmat = r_e.as_matrix()



        #test = np.diag(t_scale[0]) @ interp_rotmat[0,:,:]
        constraint_flip = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        new_traj = []
        for t in times:
            t_dict = dict()
            t_dict['scan_idx'] = t
            t_dict['obj'] = dict()
            t_dict['obj']['cad2world'] = np.identity(4)
            t_dict['obj']['cad2world'][:3, :3] = (np.diag(t_scale[t]) @ interp_rotmat[t-key_times[0],:,:])
            t_dict['obj']['cad2world'][:3, 3] = t_trans[t]
            t_dict['obj']['voxel'] = t_vox[t]
            t_dict['obj']['obj_idx'] = t_id[t]
            t_dict['obj']['compl_box'] = t_box[t]
            new_traj.append(t_dict)

        new_trajectories.append(new_traj)

    return new_trajectories


def fuse_pose_F2F(trajectories, seq_len=125, constraint=True):
    '''
    pose fusion via slurp and spline interpolation
    '''

    def get_scale(m):
        if type(m) == torch.Tensor:
            return m.norm(dim=0)
        return np.linalg.norm(m, axis=0)

    def fill_last(fill_list, exp_dim=False):
        # fill with last value
        for t_idx, tt in enumerate(fill_list):
            if tt is None:
                for i in range(t_idx - 1, -1, -1):
                    if fill_list[i] is not None:
                        if exp_dim:
                            fill_list[t_idx] = np.expand_dims(fill_list[i], axis=0)
                        else:
                            fill_list[t_idx] = fill_list[i]
                        break
        return fill_list

    def fill_last_t(fill_list, exp_dim=True):
        # fill with last value
        for t_idx, tt in enumerate(fill_list):
            if tt.sum() == 0:
                for i in range(t_idx - 1, -1, -1):
                    if fill_list[i].sum() != 0:
                        if exp_dim:
                            fill_list[t_idx] = np.expand_dims(np.squeeze(fill_list[i]), axis=0)
                        else:
                            fill_list[t_idx] = fill_list[i]
                        break
            else:
                fill_list[t_idx] = np.expand_dims(fill_list[t_idx], axis=0)

        for t_idx, tt in enumerate(fill_list):
            if len(tt.shape) == 1:
                fill_list[t_idx] = np.expand_dims(fill_list[t_idx], axis=0)

        return fill_list

    def unscale_mat(cad2world):

        c2w_cpy = np.copy(cad2world)
        rot = cad2world[:3, :3]
        scale = get_scale(rot)
        unscaled_rot = rot / scale
        c2w_cpy[:3, :3] = unscaled_rot
        return c2w_cpy

    new_trajectories = []

    times = np.arange(seq_len)
    for traj in trajectories:
        key_times = []
        key_trans = []
        key_rots = []
        t_trans = [np.zeros(3) for i in range(seq_len)]
        t_vox = [None for i in range(seq_len)]
        t_box = [None for i in range(seq_len)]
        t_id = [None for i in range(seq_len)]
        t_scale = [None for i in range(seq_len)]
        for pred in traj:
            key_rots.append(np.expand_dims(unscale_mat(pred['obj']['cad2world'][:3, :3]), axis=0))
            key_trans.append(np.expand_dims(pred['obj']['cad2world'][:3, 3], axis=0))
            key_times.append(pred['scan_idx'])
            t_trans[pred['scan_idx']] = pred['obj']['cad2world'][:3, 3]#.numpy()
            t_vox[pred['scan_idx']] = pred['obj']['obj_pc']
            t_id[pred['scan_idx']] = pred['obj']['obj_idx']
            t_box[pred['scan_idx']] = pred['obj']['obj_box']
            t_scale[pred['scan_idx']] = get_scale(pred['obj']['cad2world'][:3, :3])

        times = np.linspace(key_times[0], key_times[-1], num=key_times[-1]-key_times[0]+1).astype(np.int)
        traj_rots = np.concatenate(key_rots, axis=0)#.numpy()
        #key_trans = torch.cat(key_trans, dim=0)#.numpy()

        t_trans = np.concatenate(fill_last_t(t_trans, exp_dim=True), axis=0)
        t_trans[:, 0] = gaussian_filter1d(t_trans[:, 0], 3)
        t_trans[:, 1] = gaussian_filter1d(t_trans[:, 1], 3)
        t_trans[:, 2] = gaussian_filter1d(t_trans[:, 2], 3)
        t_vox = fill_last(t_vox)
        t_id = fill_last(t_id)
        t_box = fill_last(t_box)
        t_scale = fill_last(t_scale)

        r = R.from_matrix(traj_rots)
        slerp = Slerp(key_times, r)
        interp_rots = slerp(times)
        interp_rotmat = interp_rots.as_matrix()
        euler_rots = interp_rots.as_euler('xyz')
        euler_rots[:,-1] = gaussian_filter1d(euler_rots[:,-1], 3) #3 = sigma = standard deviation
        euler_rots[:, -1] = np.clip(euler_rots[:,-1], euler_rots[0,-1] - (0.2 * euler_rots[0,-1]), euler_rots[0,-1] + (0.2 * euler_rots[0,-1]))
        if constraint:
            euler_rots[:,0] = 0
            euler_rots[:,1] = 0
        r_e = R.from_euler('xyz', euler_rots, degrees=False)
        interp_rotmat = r_e.as_matrix()


        #test = np.diag(t_scale[0]) @ interp_rotmat[0,:,:]
        constraint_flip = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        new_traj = []
        for t in times:
            t_dict = dict()
            t_dict['scan_idx'] = t
            t_dict['obj'] = dict()
            t_dict['obj']['cad2world'] = np.identity(4)
            t_dict['obj']['cad2world'][:3, :3] = (np.diag(t_scale[t]) @ interp_rotmat[t-key_times[0],:,:])
            t_dict['obj']['cad2world'][:3, 3] = t_trans[t]
            t_dict['obj']['obj_pc'] = t_vox[t]
            t_dict['obj']['obj_idx'] = t_id[t]
            t_dict['obj']['obj_box'] = t_box[t]
            new_traj.append(t_dict)

        new_trajectories.append(new_traj)

    return new_trajectories