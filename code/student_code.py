import numpy as np
import math


def calculate_projection_matrix(points_2d, points_3d):
    """


    Args:
    -   points_2d: A numpy array of shape (N, 2)
    -   points_3d: A numpy array of shape (N, 3)

    Returns:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    """




    GivenPoints = points_2d.shape[0]

    Equations = np.zeros((GivenPoints*2,11))

    Output = np.zeros((GivenPoints*2,1))





    for i in range(GivenPoints):

        WorldX = points_3d[i][0]
        WorldY = points_3d[i][1]
        WorldZ = points_3d[i][2]

        ExtractedX = points_2d[i][0]
        ExtractedY = points_2d[i][1]



        Equations[2*i,:] = [WorldX, WorldY, WorldZ, 1, 0, 0, 0, 0, -1*ExtractedX*WorldX, -1*ExtractedX*WorldY, -1*ExtractedX*WorldZ]
        Equations[2*i+1,:] = [0, 0, 0, 0, WorldX, WorldY, WorldZ, 1, -1*ExtractedY*WorldX, -1*ExtractedY*WorldY, -1*ExtractedY*WorldZ]


        Output[2*i,0] = ExtractedX
        Output[2*i+1,0] = ExtractedY


    MatrixM = np.linalg.lstsq(Equations, Output, rcond = None)[0]

    MatrixM = np.append(MatrixM, 1)

    MatrixM = np.reshape(MatrixM, (3,4))




    M = MatrixM


    return M

def calculate_camera_center(M):
    """
    Returns the camera center matrix for a given projection matrix.

    The center of the camera C can be found by:

        C = -Q^(-1)m4

    where your project matrix M = (Q | m4).

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """

    Q = M[0:3,0:3]
    M4 = M[:,3]

    Qinv = np.linalg.inv(Q)
    Qinv = -1*Qinv


    cc = np.matmul(Qinv,M4)


    return cc

def estimate_fundamental_matrix(points_a, points_b):
    """

    Args:
    -   points_a: A numpy array of shape (N, 2) representing the 2D points in
                  image A
    -   points_b: A numpy array of shape (N, 2) representing the 2D points in
                  image B

    Returns:
    -   F: A numpy array of shape (3, 3) representing the fundamental matrix
    """


    GivenPoints = points_a.shape[0]

    ScaleMatA = np.zeros((3,3))
    OffsetMatA = np.zeros((3,3))

    ScaleMatB = np.zeros((3,3))
    OffsetMatB = np.zeros((3,3))

    TransformA = np.zeros((3,3))
    TransformB = np.zeros((3,3))

    PointsTransformA = np.zeros((GivenPoints,2))
    PointsTransformB = np.zeros((GivenPoints,2))

    TempA = np.zeros((3,1))
    TempB = np.zeros((3,1))

    Mean_a = np.mean(points_a, axis = 0)
    Mean_b = np.mean(points_b, axis = 0)

    Cu1 = Mean_a[0]
    Cv1 = Mean_a[1]

    Cu2 = Mean_b[0]
    Cv2 = Mean_b[1]

    DistanceA = 0
    DistanceB = 0

    for i in range(GivenPoints):

        DistanceA = DistanceA + (points_a[i][0] - Cu1)**2 + (points_a[i][1] - Cv1)**2
        DistanceB = DistanceB + (points_b[i][0] - Cu2)**2 + (points_b[i][1] - Cv2)**2

    DistanceAvgA = DistanceA/GivenPoints
    DistanceAvgB = DistanceB/GivenPoints

    s1 = math.sqrt(2/DistanceAvgA)
    s2 = math.sqrt(2/DistanceAvgB)

    ScaleMatA[0][0] = s1
    ScaleMatA[1][1] = s1
    ScaleMatA[2][2] = 1

    OffsetMatA[0][0] = 1
    OffsetMatA[1][1] = 1
    OffsetMatA[2][2] = 1
    OffsetMatA[0][2] = -1*Cu1
    OffsetMatA[1][2] = -1*Cv1

    ScaleMatB[0][0] = s2
    ScaleMatB[1][1] = s2
    ScaleMatB[2][2] = 1

    OffsetMatB[0][0] = 1
    OffsetMatB[1][1] = 1
    OffsetMatB[2][2] = 1
    OffsetMatB[0][2] = -1*Cu2
    OffsetMatB[1][2] = -1*Cv2

    TransformA = np.matmul(ScaleMatA, OffsetMatA)
    TransformB = np.matmul(ScaleMatB, OffsetMatB)

    for k in range(GivenPoints):

        TempA = np.matmul(TransformA,[[points_a[k][0]], [points_a[k][1]], [1]])
        PointsTransformA[k, :] = [TempA[0] , TempA[1]]

        TempB = np.matmul(TransformB,[[points_b[k][0]], [points_b[k][1]], [1]])
        PointsTransformB[k, :] = [TempB[0] , TempB[1]]

    points_a = PointsTransformA
    points_b = PointsTransformB


    MatrixF = np.ones((9,1))

    Equations = np.zeros((GivenPoints,9))

    for j in range(GivenPoints):


        u1 = points_a[j][0]
        v1 = points_a[j][1]
        u2 = points_b[j][0]
        v2 = points_b[j][1]

        Equations[j,:] = [u1*u2, v1*u2, u2, u1*v2, v2*v1, v2, u1, v1, 1]


    u, s, v = np.linalg.svd(Equations)

    v = np.transpose(v)

    TempF = v[:,8]
    TempF = np.reshape(TempF, (3,3))


    Unew, Snew, Vnew = np.linalg.svd(TempF)

    Snew[2] = 0

    Snew = [[Snew[0], 0, 0],[0, Snew[1], 0], [0, 0, Snew[2]]]

    Snew = np.asarray(Snew)

    Fnew = np.matmul(np.matmul(Unew,Snew), Vnew)

    F = np.matmul(np.matmul(np.transpose(TransformB),Fnew),TransformA)


    return F

def ransac_fundamental_matrix(matches_a, matches_b):
    """

    Args:
    -   matches_a: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image A
    -   matches_b: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_a: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image A that are inliers with
                   respect to best_F
    -   inliers_b: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image B that are inliers with
                   respect to best_F
    """



    OutlierProb = 0.6
    SamplePoints = 8
    GoodProb = 0.99
    thresh = 0.005
    CorrectMatches = 0
    FinalFundamental = np.zeros((3,3))



    N = math.log10(1 - GoodProb)/(math.log10(1-(1-OutlierProb)**SamplePoints))
    N = round(N)


    PossibleMatches = matches_a.shape[0]

    for i in range(int(N)):

        inliers_axTemp = []
        inliers_ayTemp = []
        inliers_bxTemp = []
        inliers_byTemp = []

        Matchestemp = 0

        Match8 = np.random.randint(PossibleMatches, size=8)
        FundMatrixTemp = estimate_fundamental_matrix(matches_a[Match8, :], matches_b[Match8, :])


        for j in range(PossibleMatches):



            Output = np.matmul(np.matmul([matches_b[j][0], matches_b[j][1], 1],FundMatrixTemp), [[matches_a[j][0]], [matches_a[j][1]], [1]])
            Output = abs(Output)

            if Output<thresh:

                Matchestemp = Matchestemp + 1
                inliers_axTemp.append(matches_a[j][0])
                inliers_ayTemp.append(matches_a[j][1])
                inliers_bxTemp.append(matches_b[j][0])
                inliers_byTemp.append(matches_b[j][1])

        if Matchestemp > CorrectMatches:

            CorrectMatches = Matchestemp
            FinalFundamental = FundMatrixTemp
            inliers_ax = inliers_axTemp
            inliers_ay = inliers_ayTemp
            inliers_bx = inliers_bxTemp
            inliers_by = inliers_byTemp


    inliers_ax = np.asarray(inliers_ax)
    inliers_ay = np.asarray(inliers_ay)
    inliers_bx = np.asarray(inliers_bx)
    inliers_by = np.asarray(inliers_by)

    inliers_a = np.stack((inliers_ax, inliers_ay), axis = -1)
    inliers_b = np.stack((inliers_bx, inliers_by), axis = -1)
    best_F = FinalFundamental




    return best_F, inliers_a, inliers_b
