import sys
import time
import numpy as np
import torch
from dvis import dvis
import open3d as o3d

import trimesh
import mcubes
import mathutils
from sklearn.preprocessing import minmax_scale

def visualise_gt_sequence(gt_trajectories, seq_name=None, seq_len=125, as_mesh=True, with_box=True):
    '''
    Visualise Tracking via object idx, scan as pointcloud for background, objects as voxel grids
    '''

    boxes = []
    tracklets = dict()
    for scan_idx in range(seq_len):
        for color_idx, traj in enumerate(gt_trajectories):

            # Load BG pointcloud
            if color_idx == 0 and scan_idx == 0:
                world_pc = traj[scan_idx]['obj']['world_pc']

            for frame in traj:
                if frame['scan_idx'] == scan_idx:

                    # Voxelized objects in world space
                    world_pc_obj = grid2world(frame['obj']['voxel'], frame['obj']['cad2world'],
                                              frame['obj']['compl_box'])

                    cad2world_scaled = frame['obj']['cad2world']
                    cad2world_unscaled = unscale_mat(cad2world_scaled)

                    if as_mesh:
                        #mesh = norm_vox2mesh(frame['obj']['voxel'], frame['obj']['cad2world'], box=frame['obj']['compl_box']) # todo apply scale only
                        mesh = vox2mesh(frame['obj']['voxel'], box=None)
                        # Place at frame 0
                        if scan_idx == 0: #todo if object is not in scan idx ==0 visible object is not placed maybe find all unique objects first
                            print('Placing {} in scene'. format(f'mesh_{color_idx}'))
                            dvis(mesh, fmt='mesh', c=color_idx+1, l=[0,1], name=f'obj/mesh_{color_idx}')
                            # Timeout for loading object in dvis
                            time.sleep(5)
                        dvis(cad2world_scaled, 'obj_kf', name=f'mesh_{color_idx}', t=scan_idx+1) # maybe remove scale in cad to world here
                    
                    elif scan_idx == 0:
                        dvis(world_pc_obj, vs=0.07, c=color_idx+1, t=scan_idx+1, l=[0,1], name=f'obj/{color_idx}')

                    # Bounding box placement
                    box = np.expand_dims(frame['obj']['compl_box'], axis=0)
                    # Place box at frame 0
                    if with_box and scan_idx == 0:
                        dvis(box, fmt='box', s=3, c=color_idx+1, l=[0,3], name=f'box/box_{color_idx}')
                        # Timeout for loading object in dvis
                        #time.sleep(2)
                    # Todo need relative transformation between frame 0 and frame n
                    #dvis(cad2world_unscaled, 'obj_kf', name=f'box_{color_idx}', t=scan_idx+1)
                    '''
                    WORKS BUT COLOR CANT BE MODIFIED AFTERWARDS cause placed every t 
                    if with_box:
                        dvis(box, fmt='box', s=3, c=color_idx+1, t=scan_idx+1, l=[0,3], name=f'box/box_{color_idx}')
                    '''
                    # Obj pc center
                    obj_center = world_pc_obj.mean(axis=0)
                    if color_idx+1 in tracklets:
                        tracklets[color_idx+1].append(obj_center)
                    else:
                        tracklets[color_idx+1] = [obj_center]

                    # Boxes for cropping
                    if scan_idx == 0:
                        boxes.append(frame['obj']['compl_box'])
                    break

    # Set tracklet lines
    for c_val, l_verts in tracklets.items():
        line_verts = np.concatenate(l_verts, axis=0)
        dvis(line_verts, fmt='line', s=6, c=c_val, l=[0,2], name=f'line/line_{c_val}')

    # Vis background
    world_pc = crop_pc(world_pc, boxes)
    dvis(world_pc, fmt='xyzrgb', vs=0.02, l=[0,4], vis_conf={'opacity': 0.5}, name='background') #set opacity to 0.5


    # Set title
    dvis({"title":seq_name, "track_store_path": seq_name}, 'config')
    sys.exit()

    #Load and set camera parameters
    #dvis({}, fmt='cam')

def visualise_pred_sequence(pred_trajectories, gt_trajectories, seq_name=None, seq_len=125, with_box=False, as_mesh=True):
    '''
    Visualise Tracking via object idx, scan as pointcloud for background, objects as voxel grids
    Added smoothing:
    - fused object shape in canonical space and averaged box coordinates
    '''

    boxes = []
    tracklets = dict()
    fused_shapes, fused_scales = fuse_obj_shape(pred_trajectories)

    for scan_idx in range(seq_len):
        for color_idx, traj in enumerate(pred_trajectories):
            if color_idx == 0 and scan_idx == 0:
                world_pc = gt_trajectories[0][scan_idx]['obj']['world_pc']

            norm_obj_shape = fused_shapes[color_idx]
            norm_obj_scale = fused_scales[color_idx]
            if norm_obj_shape.is_cuda:
                norm_obj_shape = norm_obj_shape.detach().cpu()
                norm_obj_scale = norm_obj_scale.detach().cpu()

            for frame in traj:
                if frame['scan_idx'] == scan_idx:

                    cad2world = rescale_mat(frame['obj']['cad2world'], norm_obj_scale)
                    world_pc_obj = grid2world(norm_obj_shape, cad2world, None, pred=True)

                    if as_mesh:
                        #mesh = prednorm_vox2mesh(norm_obj_shape.numpy(), cad2world, box=frame['obj']['compl_box']) # idea use box to crop pc in cad space, box to cad, and scale pred mesh in cad
                        mesh = vox2mesh(norm_obj_shape.numpy(), box=None)

                        # Place at frame 0
                        if scan_idx == 0:  # todo if object is not in scan idx ==0 visible object is not placed maybe find all unique objects first
                            print('Placing {} in scene'.format(f'mesh_{color_idx}'))
                            dvis(mesh, fmt='mesh', c=color_idx+1, l=[0, 1], name=f'obj/mesh_{color_idx}')
                            # Timeout for loading object in dvis
                            time.sleep(5)
                        dvis(cad2world, 'obj_kf', name=f'mesh_{color_idx}', t=scan_idx+1)

                    elif scan_idx == 0:
                        dvis(world_pc_obj, vs=0.04, c=color_idx+1, t=scan_idx+1, l=[0,1], name=f'obj/{color_idx}')

                    # Bounding box placement
                    if with_box:
                        box = np.expand_dims(frame['obj']['compl_box'], axis=0)
                        dvis(box, fmt='box', s=3, c=color_idx+1, t=scan_idx+1, l=[0,3], name=f'box/box_{color_idx}')

                    # Obj pc center
                    obj_center = world_pc_obj.mean(axis=0)
                    if color_idx+1 in tracklets:
                        tracklets[color_idx+1].append(obj_center)
                    else:
                        tracklets[color_idx+1] = [obj_center]

                    # Boxes for cropping
                    if scan_idx == 0:
                        boxes.append(frame['obj']['compl_box'])
                    break

    # Set tracklet lines
    for c_val, l_verts in tracklets.items():
        line_verts = np.concatenate(l_verts, axis=0)
        dvis(line_verts, fmt='line', s=6, c=c_val, l=[0,2], name=f'line/{c_val}')

    # Vis background
    world_pc = crop_pc(world_pc, boxes)
    dvis(world_pc, fmt='xyzrgb', vs=0.02, l=[0,4], vis_conf={'opacity': 0.5}, name='background') #set opacity to 0.5

    # Set title
    dvis({"title":seq_name, "track_store_path": seq_name}, 'config')
    sys.exit()


def visualise_pred_sequence_F2F(pred_trajectories, gt_trajectories, seq_name=None, seq_len=125, with_box=False, as_mesh=False, world_pc=None):

    boxes = []
    tracklets = dict()
    #fused_shapes, fused_scales = fuse_obj_shape(pred_trajectories)

    for scan_idx in range(seq_len):
        for color_idx, traj in enumerate(pred_trajectories):

            for frame in traj:
                if frame['scan_idx'] == scan_idx:

                    #cad2world = rescale_mat(frame['obj']['cad2world'], norm_obj_scale)
                    #world_pc_obj = grid2world(norm_obj_shape, cad2world, None, pred=True)

                    if as_mesh:
                        #mesh = prednorm_vox2mesh(norm_obj_shape.numpy(), cad2world, box=frame['obj']['compl_box']) # idea use box to crop pc in cad space, box to cad, and scale pred mesh in cad
                        mesh = vox2mesh(norm_obj_shape.numpy(), box=None)

                        # Place at frame 0
                        if scan_idx == 0:  # todo if object is not in scan idx ==0 visible object is not placed maybe find all unique objects first
                            print('Placing {} in scene'.format(f'mesh_{color_idx}'))
                            dvis(mesh, fmt='mesh', c=color_idx+1, l=[0, 1], name=f'obj/mesh_{color_idx}')
                            # Timeout for loading object in dvis
                            time.sleep(5)
                        dvis(cad2world, 'obj_kf', name=f'mesh_{color_idx}', t=scan_idx+1)

                    elif scan_idx < 25:
                        dvis(frame['obj']['obj_pc'], vs=0.04, c=color_idx+1, t=scan_idx+1, l=[0,1], name=f'obj/{color_idx}')

                    # Obj pc center
                    obj_center = frame['obj']['obj_pc'].mean(axis=0)
                    if color_idx+1 in tracklets:
                        tracklets[color_idx+1].append(obj_center)
                    else:
                        tracklets[color_idx+1] = [obj_center]

                    # Boxes for cropping
                    if scan_idx == 0:
                        boxes.append(frame['obj']['obj_box'])
                    break

    # Set tracklet lines
    for c_val, l_verts in tracklets.items():
        line_verts = np.concatenate(l_verts, axis=0)
        dvis(line_verts, fmt='line', s=6, c=c_val, l=[0,2], name=f'line/{c_val}')

    # Vis background
    world_pc = crop_pc(world_pc, boxes)
    dvis(world_pc, fmt='xyzrgb', vs=0.02, l=[0,4], vis_conf={'opacity': 0.5}, name='background') #set opacity to 0.5

    # Set title
    dvis({"title":seq_name, "track_store_path": seq_name}, 'config')
    sys.exit()

    #Load and set camera parameters
    #dvis({}, fmt='cam')

def visualise_pred_sequence_F2F_key(pred_trajectories, gt_trajectories, seq_name=None, seq_len=125, with_box=False, as_mesh=False, world_pc=None):

    # todo fuse scale, mesh objects, place all objects from frame 0

    def world2cad(pc, cad2world):
        '''
        Transfrom PointCloud from camera space to world space
        cam_pc: N points x XYZ(3)
        '''

        world2cad_T = np.linalg.inv(cad2world)
        trans = world2cad_T[:3, 3:]
        rot = world2cad_T[:3, :3]

        world_pc = np.dot(rot, pc.transpose()) + trans
        cad_pc = world_pc.transpose()

        return cad_pc

    boxes = []
    tracklets = dict()
    fused_scalerots, fused_scale, fused_shape = fuse_obj_shape_F2F(pred_trajectories)

    for color_idx, traj in enumerate(pred_trajectories):

        print('Placing {} in scene'.format(f'mesh_{color_idx}'))
        pc_cad = world2cad(traj[0]['obj']['obj_pc'], traj[0]['obj']['cad2world'])
        norm_mesh = pc2mesh(pc_cad)
        #dvis(pc_cad, vs=0.04, c=color_idx + 1, l=[0, 1], name=f'obj/mesh_{color_idx}')
        dvis(norm_mesh, fmt='mesh', c=color_idx+1, l=[0, 1], name=f'obj/mesh_{color_idx}')
        # Timeout for loading object in dvis
        time.sleep(5)

    for scan_idx in range(seq_len):
        for color_idx, traj in enumerate(pred_trajectories):
            '''
            norm_obj_shape = fused_shapes[color_idx]
            norm_obj_scale = fused_scales[color_idx]
            if norm_obj_shape.is_cuda:
                norm_obj_shape = norm_obj_shape.detach().cpu()
                norm_obj_scale = norm_obj_scale.detach().cpu()
            '''


            for frame in traj:
                if frame['scan_idx'] == scan_idx:
                    obj_scalerot = fused_scalerots[color_idx][scan_idx]

                    #cad2world = rescale_mat(frame['obj']['cad2world'], norm_obj_scale)
                    #world_pc_obj = grid2world(norm_obj_shape, cad2world, None, pred=True)

                    if as_mesh:
                        #mesh = prednorm_vox2mesh(norm_obj_shape.numpy(), cad2world, box=frame['obj']['compl_box']) # idea use box to crop pc in cad space, box to cad, and scale pred mesh in cad
                        mesh = vox2mesh(norm_obj_shape.numpy(), box=None)

                        # Place at frame 0
                        if scan_idx == 0:  # todo if object is not in scan idx ==0 visible object is not placed maybe find all unique objects first
                            print('Placing {} in scene'.format(f'mesh_{color_idx}'))
                            dvis(mesh, fmt='mesh', c=color_idx+1, l=[0, 1], name=f'obj/mesh_{color_idx}')
                            # Timeout for loading object in dvis
                            time.sleep(5)
                        dvis(cad2world, 'obj_kf', name=f'mesh_{color_idx}', t=scan_idx+1)

                    if not as_mesh:
                        ca2w = frame['obj']['cad2world']
                        scale = fused_scale[color_idx]
                        ca2w[:3,:3] =  np.diag(scale) @ obj_scalerot
                        dvis(ca2w, 'obj_kf', name=f'mesh_{color_idx}', t=scan_idx + 1)


                    # Obj pc center
                    obj_center = frame['obj']['obj_pc'].mean(axis=0)
                    if color_idx+1 in tracklets:
                        tracklets[color_idx+1].append(obj_center)
                    else:
                        tracklets[color_idx+1] = [obj_center]

                    # Boxes for cropping
                    if scan_idx == 0:
                        boxes.append(frame['obj']['obj_box'])
                    break

    # Set tracklet lines
    for c_val, l_verts in tracklets.items():
        line_verts = np.concatenate(l_verts, axis=0)
        dvis(line_verts, fmt='line', s=6, c=c_val, l=[0,2], name=f'line/{c_val}')

    # Vis background
    world_pc = crop_pc(world_pc, boxes)
    dvis(world_pc, fmt='xyzrgb', vs=0.02, l=[0,4], vis_conf={'opacity': 0.5}, name='background') #set opacity to 0.5

    # Set title
    dvis({"title":seq_name, "track_store_path": seq_name}, 'config')
    sys.exit()

    #Load and set camera parameters
    #dvis({}, fmt='cam')


def visualise_pred_sequence_office(pred_trajectories, world_pc, seq_name=None, seq_len=25, with_box=False, as_mesh=True):
    '''
    Visualise Tracking via object idx, scan as pointcloud for background, objects as voxel grids
    Setting for office dataset
    '''

    boxes = []
    tracklets = dict()
    fused_shapes, fused_scales = fuse_obj_shape_office(pred_trajectories)
    dvis_flip = np.diag([1, -1, -1, 1])

    # Place predicted meshes
    for color_idx, norm_obj_shape in enumerate(fused_shapes):
        if norm_obj_shape.is_cuda:
            norm_obj_shape = norm_obj_shape.detach().cpu()
        mesh = vox2mesh(norm_obj_shape.numpy(), box=None)
        print('Placing {} in scene'.format(f'mesh_{color_idx}'))
        dvis(mesh, fmt='mesh', c=color_idx + 1, l=[0, 1], name=f'obj/mesh_{color_idx}')
        # Timeout for loading object in dvis
        time.sleep(5)


    for scan_idx in range(seq_len):
        for color_idx, traj in enumerate(pred_trajectories):
            norm_obj_shape = fused_shapes[color_idx]
            norm_obj_scale = fused_scales[color_idx]
            if norm_obj_shape.is_cuda:
                norm_obj_shape = norm_obj_shape.detach().cpu()
                norm_obj_scale = norm_obj_scale.detach().cpu()
            for frame in traj:
                if frame['scan_idx'] == scan_idx:

                    # Voxelized objects place on layer 1
                    cad2world = rescale_mat(frame['obj']['cad2world'], norm_obj_scale)
                    world_pc_obj = grid2world(norm_obj_shape, cad2world, None, pred=True)

                    if as_mesh:
                        dvis(cad2world@dvis_flip, 'obj_kf', name=f'mesh_{color_idx}', t=scan_idx + 1)
                    elif scan_idx == 0:
                        dvis(world_pc_obj, vs=0.07, c=color_idx + 1, t=scan_idx+1, l=[0, 1], name=f'obj/{color_idx}')
                    '''
                        #mesh = norm_vox2mesh(norm_obj_shape.numpy(), cad2world, box=None)
                        mesh = vox2mesh(norm_obj_shape.numpy(), box=None)
                        if scan_idx == 0: #todo if object is not in scan idx ==0 visible object is not placed maybe find all unique objects first
                            print('Placing {} in scene'. format(f'mesh_{color_idx}'))
                            dvis(mesh, fmt='mesh', c=color_idx+1, l=[0,1], name=f'obj/mesh_{color_idx}')
                            # Timeout for loading object in dvis
                            time.sleep(5)
                    '''

                    # Bounding box placement
                    if with_box:
                        box = np.expand_dims(frame['obj']['compl_box'], axis=0)
                        dvis(box, fmt='box', s=3, c=color_idx+1, t=scan_idx+1, l=[0,3], name=f'box/box_{color_idx}')

                    # Obj pc center
                    obj_center = world_pc_obj.mean(axis=0)
                    if color_idx+1 in tracklets:
                        tracklets[color_idx+1].append(obj_center)
                    else:
                        tracklets[color_idx+1] = [obj_center]

                    # Boxes for cropping
                    if scan_idx == 0:
                        boxes.append(frame['obj']['compl_box'])
                    break


    # Set tracklet lines
    '''
    for c_val, l_verts in tracklets.items():
        line_verts = np.concatenate(l_verts, axis=0)
        dvis(line_verts, fmt='line', s=6, c=c_val, l=[0,2], name=f'line/{c_val}')
    '''


    # Vis background
    world_pc = crop_pc(world_pc, boxes)
    dvis(world_pc, fmt='xyzrgb', vs=0.1, l=[0, 4], vis_conf={'opacity': 0.35}, name='background')  # set opacity to 0.5

    # Set title
    dvis({"title":seq_name, "track_store_path": seq_name}, 'config')
    sys.exit()

def visualise_gt_sequence_office(world_pc, seq_name=None, seq_len=25):
    '''
    Visualise Tracking via object idx, scan as pointcloud for background
    For GT requires world pc as list with n entries for n frames
    '''

    for scan_idx in range(seq_len):
        dvis(world_pc[scan_idx], fmt='xyzrgb', vs=0.2, t=scan_idx+1, vis_conf={'opacity': 1},
             name='background')  # set opacity to 0.5

    # Set title
    dvis({"title":seq_name}, 'config')

## HELPER FCTS ---------------------------------------------------------------------------------------------------------
def grid2world(voxel_grid, cad2world, box, pred=False):
    '''
    Transform voxel obj to world space
    '''

    if not pred:
        if type(voxel_grid) == np.ndarray:
            nonzero_inds = np.nonzero(torch.from_numpy(voxel_grid))[:-1]
        else:
            nonzero_inds = np.nonzero(voxel_grid)[:-1]
    else:
        nonzero_inds = np.nonzero(voxel_grid)[:-1]
    points = nonzero_inds / 31 - 0.5
    if points.is_cuda:
        points = points.detach().cpu().numpy()
    else:
        points = points.numpy()
    #points[:, 1] -= points[:, 1].min()  # CAD space y is shifted up to start at 0

    # Cad2World
    world_pc = cad2world[:3,:3] @ points.transpose() + np.expand_dims(cad2world[:3,3], axis=-1)
    world_pc = world_pc.T

    if type(world_pc) == np.ndarray:
        scaled_pc = world_pc.copy()
    else:
        scaled_pc = world_pc.numpy().copy()
    if box is not None:
        scaled_pc[:,0] = minmax_scale(scaled_pc[:, 0], feature_range=(box[0], box[3]))
        scaled_pc[:,1] = minmax_scale(scaled_pc[:, 1], feature_range=(box[1], box[4]))
        scaled_pc[:,2] = minmax_scale(scaled_pc[:, 2], feature_range=(box[2], box[5]))

    return scaled_pc

def crop_pc(pc, boxes):
    '''
    Crop points which voxels will be placed at
    '''
    idxs = []
    for keep_idx, pt in enumerate(pc):
        keep = True
        for box in boxes:
            if pt[0] >= box[0]-0.01 and pt[0] <= box[3]+0.01 and pt[1] >= box[1]-0.01 and pt[1] <= box[4]+0.01 and pt[2] >= box[2]-0.01 and pt[2] <= box[5]+0.01:
                keep = False
        if keep == True:
            idxs.append(keep_idx)

    cropped_pc = pc[idxs,:]

    return cropped_pc

def pc2mesh(pc, box=None):
    '''
    Run marching cubes over voxel grid and set in world space
    '''

    def pc2vox(pc):
        RANGE_START = pc.min()
        RANGE_END = pc.max()
        VOXEL_K = 31

        # shift points to 0-based:
        zero_based_points = pc - RANGE_START

        # convert points to [0, 1) fraction of range
        fractional_points = zero_based_points / (RANGE_END - RANGE_START)

        # project points into voxel space: [0, k)
        voxelspace_points = fractional_points * VOXEL_K

        # convert voxel space to voxel indices (truncate decimals: 0.1 -> 0)
        voxel_indices = voxelspace_points.astype(int)
        x = voxel_indices[:, 0].astype(np.int32)
        y = voxel_indices[:, 1].astype(np.int32)
        z = voxel_indices[:, 2].astype(np.int32)
        rescale_ = np.zeros((32, 32, 32))
        rescale_[x, y, z] = 1
        return rescale_

    if type(pc) == torch.Tensor:
        pc = pc.numpy()

    vox = pc2vox(pc)

    vertices, triangles = mcubes.marching_cubes(vox, 0)
    # Verticies to CAD space before applying transform
    vertices = vertices / 31 - 0.5

    if box is not None:
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    vertices = mesh.vertices


    if box is not None:
        vertices[:,0] = minmax_scale(vertices[:, 0], feature_range=(box[0], box[3]))
        vertices[:,1] = minmax_scale(vertices[:, 1], feature_range=(box[1], box[4]))
        vertices[:,2] = minmax_scale(vertices[:, 2], feature_range=(box[2], box[5]))

        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    return mesh


def prednorm_vox2mesh(vox, cad2world, box=None):
    '''
    Run marching cubes over voxel grid and set in world space
    '''
    if type(vox) == torch.Tensor:
        vox = vox.numpy()
    vertices, triangles = mcubes.marching_cubes(vox, 0)
    # Verticies to CAD space before applying transform
    vertices = vertices / 31 - 0.5

    scale = get_scale(cad2world[:3, :3])
    scale_mat = torch.diag(scale)

    world2cad = np.linalg.inv(cad2world.numpy())
    box = world2cad @ box_pts

    if box is not None:
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    mesh.apply_transform(cad2world)

    vertices = mesh.vertices


    if box is not None:
        vertices[:,0] = minmax_scale(vertices[:, 0], feature_range=(box[0], box[3]))
        vertices[:,1] = minmax_scale(vertices[:, 1], feature_range=(box[1], box[4]))
        vertices[:,2] = minmax_scale(vertices[:, 2], feature_range=(box[2], box[5]))

        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    return mesh


def norm_vox2mesh(vox, cad2world, box=None):
    '''
    Run marching cubes over voxel grid and set in world space
    '''
    if type(vox) == torch.Tensor:
        vox = vox.numpy()
    vertices, triangles = mcubes.marching_cubes(vox, 0)
    # Verticies to CAD space before applying transform
    vertices = vertices / 31 - 0.5

    if box is not None:
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    mesh.apply_transform(cad2world)

    vertices = mesh.vertices


    if box is not None:
        vertices[:,0] = minmax_scale(vertices[:, 0], feature_range=(box[0], box[3]))
        vertices[:,1] = minmax_scale(vertices[:, 1], feature_range=(box[1], box[4]))
        vertices[:,2] = minmax_scale(vertices[:, 2], feature_range=(box[2], box[5]))

        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    return mesh

def vox2mesh(vox, box=None):
    '''
    Run marching cubes over voxel grid and set in world space
    '''
    if type(vox) == torch.Tensor:
        vox = vox.numpy()
    vertices, triangles = mcubes.marching_cubes(vox, 0)
    # Verticies to CAD space before applying transform
    vertices = vertices / 31 - 0.5

    if box is not None:
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    vertices = mesh.vertices


    if box is not None:
        vertices[:,0] = minmax_scale(vertices[:, 0], feature_range=(box[0], box[3]))
        vertices[:,1] = minmax_scale(vertices[:, 1], feature_range=(box[1], box[4]))
        vertices[:,2] = minmax_scale(vertices[:, 2], feature_range=(box[2], box[5]))

        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    return mesh

def fuse_obj_shape(pred_trajectories):
    '''
    fuse object shape by averaging over all predictions
    fuse object scale by averaging over all predictions
    '''

    fused_shapes = [[] for i in range(len(pred_trajectories))]
    fused_scales = [[] for i in range(len(pred_trajectories))]
    for traj_idx, traj in enumerate(pred_trajectories):
        shape = []
        scale = []
        for f_pred in traj:
            shape.append(torch.unsqueeze(f_pred['obj']['voxel'], dim=0))
            scale.append(get_scale(f_pred['obj']['cad2world'][:3,:3])[0])
        shape = torch.squeeze(torch.mean(torch.cat(shape, dim=0), dim=0), dim=0)  # 32 x 32 x 32 averages
        # Binarize shape again
        shape[shape >= 0.5] = 1
        shape[shape < 0.5] = 0
        fused_shapes[traj_idx] = shape
        fused_scales[traj_idx] = torch.mean(torch.tensor(scale))

    return fused_shapes, fused_scales

def fuse_obj_shape_F2F(pred_trajectories, seq_len=125):
    '''
    fuse object shape by averaging over all predictions
    fuse object scale by averaging over all predictions
    '''

    fused_scale = [None for i in range(len(pred_trajectories))]
    fused_shape = [None for i in range(len(pred_trajectories))]
    fused_rots = [[None for n in range(seq_len)] for i in range(len(pred_trajectories))]

    for traj_idx, traj in enumerate(pred_trajectories):
        scales = []
        for t_idx, f_pred in enumerate(traj):
            if t_idx == 0:
                fused_shape[traj_idx] = f_pred['obj']['obj_pc']
            scale = np.expand_dims(get_scale(f_pred['obj']['cad2world'][:3, :3]), axis=0)
            scales.append(scale)

            scale_rot = traj[0]['obj']['cad2world'][:3, :3] / get_scale(traj[0]['obj']['cad2world'][:3, :3])
            rot = mathutils.Matrix((scale_rot)).to_euler()
            if t_idx == 0:
                sign_vals = np.sign(np.array(mathutils.Matrix((scale_rot)).to_euler()))
            else:
                flipped = []
                for ax_idx, axis in enumerate(np.array(rot)):
                    if np.sign(axis) != np.sign(sign_vals[ax_idx]):
                        flipped.append(axis*(-1))
                    else:
                        flipped.append(axis)
                scale_rot = mathutils.Euler((flipped)).to_matrix()
            fused_rots[traj_idx][f_pred['scan_idx']] = np.array(scale_rot)
        scales = np.concatenate(scales, axis=0).mean(axis=0)
        fused_scale[traj_idx] = scales

    return fused_rots, fused_scale, fused_shape

def fuse_obj_shape_office(pred_trajectories):
    '''
    fuse object shape by averaging over all predictions
    fuse object scale by averaging over all predictions
    '''

    fused_shapes = [[] for i in range(len(pred_trajectories))]
    fused_scales = [[] for i in range(len(pred_trajectories))]
    for traj_idx, traj in enumerate(pred_trajectories):
        shape = []
        scale = []
        for f_pred in traj:
            shape.append(torch.unsqueeze(f_pred['obj']['voxel'], dim=0))
            scale.append(get_scale(f_pred['obj']['cad2world'][:3,:3])[0])
        shape = torch.squeeze(torch.mean(torch.cat(shape, dim=0), dim=0), dim=0)  # 32 x 32 x 32 averages
        # Binarize shape again
        shape[shape >= 0.5] = 1
        shape[shape < 0.5] = 0
        fused_shapes[traj_idx] = shape
        fused_scales[traj_idx] = torch.mean(torch.tensor(scale))

    return fused_shapes, fused_scales

def rescale_mat(cad2world, norm_scale):
    '''
    Rescale cad2world matrix with fused scale parameter
    '''
    rot = cad2world[:3,:3]
    unscaled_rot = rot / get_scale(rot)
    scaled_rot = torch.diag(torch.tensor([norm_scale, norm_scale, norm_scale])) @ unscaled_rot
    cad2world[:3, :3] = scaled_rot
    return cad2world

def unscale_mat(cad2world):
    '''
    Unscale cad2world matrix
    '''
    c2w_cpy = torch.clone(cad2world)
    rot = cad2world[:3,:3]
    scale = get_scale(rot)
    unscaled_rot = rot / scale
    c2w_cpy[:3, :3] = unscaled_rot
    return c2w_cpy

def get_scale(m):
    if type(m) == torch.Tensor:
        return m.norm(dim=0)
    return np.linalg.norm(m, axis=0)

