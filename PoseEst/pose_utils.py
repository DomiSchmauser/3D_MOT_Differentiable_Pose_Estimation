import sys
import numpy as np


def evaluateModel(OutTransform, SourceHom, TargetHom, PassThreshold):

    Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
    Residual = np.linalg.norm(ResidualVec)
    InlierIdx = np.where(ResidualVec < PassThreshold)
    nInliers = np.count_nonzero(InlierIdx)
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
    OutTransform[:3, :3] = ScaleMatrix @ Rotation # todo check if T is correct
    OutTransform[:3, 3] = Translation

    return Scales, Rotation, Translation, OutTransform

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
