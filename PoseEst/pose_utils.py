import sys
import numpy as np
import torch


def evaluateModel(OutTransform, SourceHom, TargetHom, PassThreshold):

    Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
    Residual = np.linalg.norm(ResidualVec)
    InlierIdx = np.where(ResidualVec < PassThreshold)
    nInliers = np.count_nonzero(InlierIdx)
    InlierRatio = nInliers / SourceHom.shape[1]

    return Residual, InlierRatio, InlierIdx[0]

def evaluateModel_torch(OutTransform, SourceHom, TargetHom, PassThreshold):

    Diff = TargetHom - torch.matmul(OutTransform, SourceHom)
    ResidualVec = torch.linalg.norm(Diff[:3, :], dim=0)
    Residual = torch.linalg.norm(ResidualVec)
    InlierIdx = torch.where(ResidualVec < PassThreshold)
    nInliers = torch.count_nonzero(InlierIdx[0])
    InlierRatio = nInliers / SourceHom.shape[1]

    return Residual, InlierRatio, InlierIdx[0]

def estimateSimilarityUmeyama(SourceHom, TargetHom):
    '''
    Procrustes analysis for pose fitting
    SourceHom: Pointcloud from NOCS map
    TargetHom: Depth pointcloud equals GT
    '''

    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    TargetCentroid = np.mean(TargetHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]

    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    CenteredTarget = TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()

    CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints

    if np.isnan(CovMatrix).any():
        print('nPoints:', nPoints)
        print(SourceHom.shape)
        print(TargetHom.shape)
        raise RuntimeError('There are NANs in the input.')

    U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]

    Rotation = np.matmul(U, Vh).T

    varP = np.var(SourceHom[:3, :], axis=1).sum()
    if varP * np.sum(D) != 0:
        ScaleFact = 1/varP * np.sum(D)  # scale factor
    else:
        ScaleFact = 1  # scale factor set to 1 since otherwise division by 0

    Scales = np.array([ScaleFact, ScaleFact, ScaleFact])
    ScaleMatrix = np.diag(Scales)

    Translation = TargetHom[:3, :].mean(axis=1) - SourceHom[:3, :].mean(axis=1).dot(ScaleFact*Rotation)

    OutTransform = np.identity(4)
    OutTransform[:3, :3] = ScaleMatrix @ Rotation
    OutTransform[:3, 3] = Translation

    return Scales, Rotation, Translation, OutTransform


def umeyama_torch(from_points, to_points):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert len(from_points.shape) == 2, \
        "from_points must be a m x n array"
    assert from_points.shape == to_points.shape, \
        "from_points and to_points must have the same shape"

    N, m = from_points.shape # N x 3

    if m == 4:
        from_points = from_points[:,:3]
        to_points = to_points[:,:3]
        m -= 1

    mean_from = from_points.mean(axis=0)
    mean_to = to_points.mean(axis=0)

    delta_from = from_points - mean_from  # N x m
    delta_to = to_points - mean_to  # N x m

    sigma_from = (delta_from * delta_from).sum(axis=1).mean()
    sigma_to = (delta_to * delta_to).sum(axis=1).mean()

    cov_matrix = delta_to.T @ (delta_from) / N
    # with timer.Timer('svd'):
    # svd = GESVD()
    U, d, V = torch.svd(cov_matrix)  # svd(cov_matrix) #
    V_t = V.T
    cov_rank = torch.linalg.matrix_rank(cov_matrix)
    S = torch.eye(m).to(from_points)

    if cov_rank >= m - 1 and torch.det(cov_matrix) < 0:
        S[m - 1, m - 1] = -1
    elif cov_rank < m - 1:
        # raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))
        return S, 1 / sigma_from, mean_to - 1 / sigma_from * mean_from

    R = U @ S @ V_t
    c = (d * S.diag()).sum() / sigma_from
    t = mean_to - (c * R) @ mean_from

    Scales = torch.tensor([c, c, c])
    ScaleMatrix = torch.diag(Scales).to(device)

    OutTransform = torch.eye(4).to(device)
    OutTransform[:3, :3] = ScaleMatrix @ R
    OutTransform[:3, 3] = t

    return R, c, t, OutTransform

def getRANSACInliers(SourceHom, TargetHom, MaxIterations=100, PassThreshold=200, StopThreshold=1):
    '''
    RANSAC Outlier Removal
    '''

    BestResidual = 1e10
    BestInlierRatio = 0
    BestInlierIdx = np.arange(SourceHom.shape[1])
    for i in range(0, MaxIterations):
        # Pick 10 random (but corresponding) points from source and target
        RandIdx = np.random.randint(SourceHom.shape[1], size=10)
        _, _, _, OutTransform = estimateSimilarityUmeyama(SourceHom[:, RandIdx], TargetHom[:, RandIdx])
        Residual, InlierRatio, InlierIdx = evaluateModel(OutTransform, SourceHom, TargetHom, PassThreshold)
        if Residual < BestResidual:
            BestResidual = Residual
            BestInlierRatio = InlierRatio
            BestInlierIdx = InlierIdx
        if BestResidual < StopThreshold:
            break

    return SourceHom[:, BestInlierIdx], TargetHom[:, BestInlierIdx], BestInlierRatio

def getRANSACInliers_torch(SourceHom, TargetHom, MaxIterations=100, PassThreshold=200, StopThreshold=1, device=None):
    '''
    RANSAC Outlier Removal
    '''

    BestResidual = 1e10
    BestInlierRatio = 0
    BestInlierIdx = torch.arange(SourceHom.shape[1]).to(device)
    for i in range(0, MaxIterations):
        # Pick 10 random (but corresponding) points from source and target
        RandIdx = torch.from_numpy(np.random.randint(SourceHom.shape[1], size=10)).to(device)
        _, _, _, OutTransform = umeyama_torch(SourceHom[:, RandIdx].T, TargetHom[:, RandIdx].T)
        Residual, InlierRatio, InlierIdx = evaluateModel_torch(OutTransform, SourceHom, TargetHom, PassThreshold)
        if Residual < BestResidual:
            BestResidual = Residual
            BestInlierRatio = InlierRatio
            BestInlierIdx = InlierIdx
        if BestResidual < StopThreshold:
            break

    return SourceHom[:, BestInlierIdx], TargetHom[:, BestInlierIdx], BestInlierRatio

def estimateSimilarityTransform(source: np.array, target: np.array, verbose=False, ratio_adapt = 1):
    SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    TargetHom = np.transpose(np.hstack([target, np.ones([source.shape[0], 1])]))

    # Auto-parameter selection based on source-target heuristics
    TargetNorm = np.mean(np.linalg.norm(target, axis=1))
    SourceNorm = np.mean(np.linalg.norm(source, axis=1))
    RatioTS = (TargetNorm / SourceNorm)
    RatioST = (SourceNorm / TargetNorm)
    PassT = RatioST*ratio_adapt if(RatioST>RatioTS) else RatioTS*ratio_adapt
    StopT = PassT / 100
    nIter = 100
    if verbose:
        print('Pass threshold: ', PassT)
        print('Stop threshold: ', StopT)
        print('Number of iterations: ', nIter)

    SourceInliersHom, TargetInliersHom, BestInlierRatio = getRANSACInliers(SourceHom, TargetHom, MaxIterations=nIter, PassThreshold=PassT, StopThreshold=StopT)

    if(BestInlierRatio < 0.1):
        print('[ WARN ] - Something is wrong. Small BestInlierRatio: ', BestInlierRatio)
        return None, None, None, None

    Scales, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(SourceInliersHom, TargetInliersHom)

    if verbose:
        print('BestInlierRatio:', BestInlierRatio)
        print('Rotation:\n', Rotation)
        print('Translation:\n', Translation)
        print('Scales:', Scales)

    return Scales, Rotation, Translation, OutTransform

def estimateSimilarityTransform_torch(source: torch.tensor, target: torch.tensor, verbose=False, ratio_adapt = 1, device=None):
    SourceHom = torch.hstack([source, torch.ones([source.shape[0], 1]).to(device)]).T
    TargetHom = torch.hstack([target, torch.ones([source.shape[0], 1]).to(device)]).T

    # Auto-parameter selection based on source-target heuristics
    TargetNorm = torch.mean(torch.linalg.norm(target, dim=1))
    SourceNorm = torch.mean(torch.linalg.norm(source, dim=1))
    RatioTS = (TargetNorm / SourceNorm)
    RatioST = (SourceNorm / TargetNorm)
    PassT = RatioST*ratio_adapt if(RatioST>RatioTS) else RatioTS*ratio_adapt
    StopT = PassT / 100
    nIter = 100
    if verbose:
        print('Pass threshold: ', PassT)
        print('Stop threshold: ', StopT)
        print('Number of iterations: ', nIter)

    SourceInliersHom, TargetInliersHom, BestInlierRatio = getRANSACInliers_torch(
        SourceHom, TargetHom, MaxIterations=nIter, PassThreshold=PassT, StopThreshold=StopT, device=device)

    if(BestInlierRatio < 0.1):
        print('[ WARN ] - Something is wrong. Small BestInlierRatio: ', BestInlierRatio)
        return None, None, None, None

    Scales, Rotation, Translation, OutTransform = umeyama_torch(SourceInliersHom.T, TargetInliersHom.T)

    if verbose:
        print('BestInlierRatio:', BestInlierRatio)
        print('Rotation:\n', Rotation)
        print('Translation:\n', Translation)
        print('Scales:', Scales)

    return Scales, Rotation, Translation, OutTransform
