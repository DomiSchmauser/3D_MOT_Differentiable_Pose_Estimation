import torch
import numpy as np
from scipy import ndimage
# from skimage.measure import block_reduce
from utils.libvoxelize.voxelize import voxelize_mesh_
from utils.libmesh import check_mesh_contains


class VoxelGrid:
    def __init__(self, data, loc=(0., 0., 0.), scale=1):
        assert(data.shape[0] == data.shape[1] == data.shape[2])
        data = np.asarray(data, dtype=np.bool)
        loc = np.asarray(loc)
        self.data = data
        self.loc = loc
        self.scale = scale

    @property
    def resolution(self):
        assert(self.data.shape[0] == self.data.shape[1] == self.data.shape[2])
        return self.data.shape[0]

    def contains(self, points):
        nx = self.resolution

        # Rescale bounding box to [-0.5, 0.5]^3
        points = (points - self.loc) / self.scale
        # Discretize points to [0, nx-1]^3
        points_i = ((points + 0.5) * nx).astype(np.int32)
        # i1, i2, i3 have sizes (batch_size, T)
        i1, i2, i3 = points_i[..., 0],  points_i[..., 1],  points_i[..., 2]
        # Only use indices inside bounding box
        mask = (
            (i1 >= 0) & (i2 >= 0) & (i3 >= 0)
            & (nx > i1) & (nx > i2) & (nx > i3)
        )
        # Prevent out of bounds error
        i1 = i1[mask]
        i2 = i2[mask]
        i3 = i3[mask]

        # Compute values, default value outside box is 0
        occ = np.zeros(points.shape[:-1], dtype=np.bool)
        occ[mask] = self.data[i1, i2, i3]

        return occ


def voxelize_ray(mesh, resolution):
    occ_surface = voxelize_surface(mesh, resolution)
    # TODO: use surface voxels here?
    occ_interior = voxelize_interior(mesh, resolution)
    occ = (occ_interior | occ_surface)
    return occ


def voxelize_fill(mesh, resolution):
    bounds = mesh.bounds
    if (np.abs(bounds) >= 0.5).any():
        raise ValueError('voxelize fill is only supported if mesh is inside [-0.5, 0.5]^3/')

    occ = voxelize_surface(mesh, resolution)
    occ = ndimage.morphology.binary_fill_holes(occ)
    return occ


def voxelize_surface(mesh, resolution):
    vertices = mesh.vertices
    faces = mesh.faces

    vertices = (vertices + 0.5) * resolution # in range[0,32]
    face_loc = vertices[faces]
    occ = np.full((resolution,) * 3, 0, dtype=np.int32)
    face_loc = face_loc.astype(np.float32)

    voxelize_mesh_(occ, face_loc)
    occ = (occ != 0)

    return occ


def voxelize_interior(mesh, resolution):
    shape = (resolution,) * 3
    bb_min = (0.5,) * 3
    bb_max = (resolution - 0.5,) * 3
    # Create points. Add noise to break symmetry
    points = make_3d_grid(bb_min, bb_max, shape=shape).numpy()
    points = points + 0.1 * (np.random.rand(*points.shape) - 0.5)
    points = (points / resolution - 0.5)
    occ = check_mesh_contains(mesh, points)
    occ = occ.reshape(shape)

    return occ


def check_voxel_occupied(occupancy_grid):
    occ = occupancy_grid

    occupied = (
        occ[..., :-1, :-1, :-1]
        & occ[..., :-1, :-1, 1:]
        & occ[..., :-1, 1:, :-1]
        & occ[..., :-1, 1:, 1:]
        & occ[..., 1:, :-1, :-1]
        & occ[..., 1:, :-1, 1:]
        & occ[..., 1:, 1:, :-1]
        & occ[..., 1:, 1:, 1:]
    )
    return occupied


def check_voxel_unoccupied(occupancy_grid):
    occ = occupancy_grid

    unoccupied = ~(
        occ[..., :-1, :-1, :-1]
        | occ[..., :-1, :-1, 1:]
        | occ[..., :-1, 1:, :-1]
        | occ[..., :-1, 1:, 1:]
        | occ[..., 1:, :-1, :-1]
        | occ[..., 1:, :-1, 1:]
        | occ[..., 1:, 1:, :-1]
        | occ[..., 1:, 1:, 1:]
    )
    return unoccupied


def check_voxel_boundary(occupancy_grid):
    occupied = check_voxel_occupied(occupancy_grid)
    unoccupied = check_voxel_unoccupied(occupancy_grid)
    return ~occupied & ~unoccupied


def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p
