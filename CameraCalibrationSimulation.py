import math
import numpy as np
import fourto
import cv2
from scipy.optimize import lsq_linear

DefaultToBASE_translation = []
DefaultToBASE_rotation = []
MarketToCamera_translation = []
MarketToCamera_rotation = []
DefaultToBASE = []
MarketToCamera = []
I=np.eye(3)
for i in range(0,17):
    matrix = np.zeros((1, 3))
    matrix1 = np.zeros((1, 4))
    matrix2 = np.zeros((3, 3))
    matrix3 = np.zeros((4, 4))
    DefaultToBASE_translation.append(matrix)
    DefaultToBASE_rotation.append(matrix1)
    MarketToCamera_translation.append(matrix)
    MarketToCamera_rotation.append(matrix1)
    DefaultToBASE.append(matrix3)
    MarketToCamera.append(matrix3)
DefaultToBASE_translation[0]=np.array([+0.48,+0.17,+0.88])
DefaultToBASE_rotation[0]=np.array([+0.00,-0.83,-0.00,-0.56])
MarketToCamera_translation[0]=np.array([+0.04,-0.38,+0.48])
MarketToCamera_rotation[0]=np.array([-0.00,+0.83,+0.00,-0.56])

DefaultToBASE_translation[1]=np.array([+0.51,+0.02,+0.96])
DefaultToBASE_rotation[1]=np.array([+0.00,-0.83,-0.00,-0.56])
MarketToCamera_translation[1]=np.array([+0.12,-0.23,+0.49])
MarketToCamera_rotation[1]=np.array([-0.00,+0.83,+0.00,-0.56])

DefaultToBASE_translation[2]=np.array([+0.62,+0.04, +0.79])
DefaultToBASE_rotation[2]=np.array([+0.00,-0.83,-0.00,-0.56])
MarketToCamera_translation[2]=np.array([+0.00, -0.25, +0.32])
MarketToCamera_rotation[2]=np.array([-0.00,+0.83,+0.00, -0.56])

DefaultToBASE_translation[3]=[+0.66,-0.06,+0.81]
DefaultToBASE_rotation[3]=[+0.00,-0.83,-0.00,-0.56]
MarketToCamera_translation[3]=[+0.04,-0.15, +0.30]
MarketToCamera_rotation[3]=[-0.00,+0.83,+0.00,-0.56]

DefaultToBASE_translation[4]=[+0.63,+0.08,+0.64]
DefaultToBASE_rotation[4]=[+0.00,-0.83,-0.00,-0.56]
MarketToCamera_translation[4]=[-0.13,-0.29,+0.26]
MarketToCamera_rotation[4]=[-0.00,+0.83,+0.00,-0.56]

DefaultToBASE_translation[5]=[+0.59,+0.18,+0.74]
DefaultToBASE_rotation[5]=[+0.00,-0.83,-0.00,-0.56]
MarketToCamera_translation[5]=[-0.05,-0.39,+0.33]
MarketToCamera_rotation[5]=[-0.00,+0.83,+0.00,-0.56]

DefaultToBASE_translation[6]=[+0.54, +0.23, +0.84]
DefaultToBASE_rotation[6]=[+0.00,-0.83,-0.00,-0.56]
MarketToCamera_translation[6]=[+0.03,-0.44, +0.42]
MarketToCamera_rotation[6]=[-0.00,+0.83,+0.00,-0.56]

DefaultToBASE_translation[7]=[+0.48,+0.22, +0.93]
DefaultToBASE_rotation[7]=[-0.09,-0.68,+0.45,-0.57]
MarketToCamera_translation[7]=[+0.15,+0.21, +0.36]
MarketToCamera_rotation[7]=[+0.09,+0.68,-0.45,-0.57]

DefaultToBASE_translation[8]=[+0.43,+0.05, +1.04]
DefaultToBASE_rotation[8]=[+0.09,+0.68,-0.45,+0.57]
MarketToCamera_translation[8]=[+0.16, +0.40, +0.27]
MarketToCamera_rotation[8]=[-0.09,-0.68,+0.45,+0.57]

DefaultToBASE_translation[9]=[+0.50,+0.02, +0.94]
DefaultToBASE_rotation[9]=[-0.12,+0.58,-0.01,+0.80]
MarketToCamera_translation[9]=[+0.45,-0.20, +0.20]
MarketToCamera_rotation[9]=[+0.12,-0.58,+0.01,+0.80]

DefaultToBASE_translation[10]=[+0.47,-0.25, +0.95]
DefaultToBASE_rotation[10]=[+0.12,-0.58,+0.01,-0.80]
MarketToCamera_translation[10]=[+0.43,+0.06,+0.27]
MarketToCamera_rotation[10]=[-0.12,+0.58,-0.01,-0.80]

DefaultToBASE_translation[11]=[+0.55,-0.36,+0.84]
DefaultToBASE_rotation[11]=[-0.12,+0.58,-0.01,+0.80]
MarketToCamera_translation[11]=[+0.29,+0.15,+0.25]
MarketToCamera_rotation[11]=[+0.12,-0.58,+0.01,+0.80]

DefaultToBASE_translation[12]=[+0.48,-0.41, +0.93]
DefaultToBASE_rotation[12]=[+0.12,-0.58,+0.01,-0.80]
MarketToCamera_translation[12]=[+0.38,+0.21, +0.30]
MarketToCamera_rotation[12]=[-0.12, +0.58,-0.01, -0.80]

DefaultToBASE_translation[13]=[+0.54,+0.43, +0.91]
DefaultToBASE_rotation[13]=[-0.12,+0.58,-0.01, +0.80]
MarketToCamera_translation[13]=[+0.47,-0.61, +0.10]
MarketToCamera_rotation[13]=[+0.12,-0.58,+0.01, +0.80]

DefaultToBASE_translation[14]=[+0.62,+0.53,+0.64]
DefaultToBASE_rotation[14]=[-0.12,+0.58,-0.01, +0.80]
MarketToCamera_translation[14]=[+0.21,-0.75,+0.08]
MarketToCamera_rotation[14]=[+0.12,-0.58,+0.01, +0.80]

DefaultToBASE_translation[15]=[+0.72, +0.33,+0.61]
DefaultToBASE_rotation[15]=[-0.12,+0.58,-0.01, +0.80]
MarketToCamera_translation[15]=[+0.12,-0.54, +0.03]
MarketToCamera_rotation[15]=[+0.12,-0.58, +0.01, +0.80]

DefaultToBASE_translation[16]=[+0.79,+0.08,+0.70]
DefaultToBASE_rotation[16]=[-0.12,+0.58,-0.01, +0.80]
MarketToCamera_translation[16]=[+0.14,-0.28,-0.01]
MarketToCamera_rotation[16]=[+0.12,-0.58, +0.01, +0.80]

for i in range(0,17):
    DefaultToBASE_rotation[i]=fourto.quaternion2rot(DefaultToBASE_rotation[i])
    DefaultToBASE[i]=fourto.create_4x4_matrix(DefaultToBASE_rotation[i], DefaultToBASE_translation[i])
    MarketToCamera_rotation[i]=fourto.quaternion2rot(MarketToCamera_rotation[i])
    MarketToCamera[i]=fourto.create_4x4_matrix(MarketToCamera_rotation[i], MarketToCamera_translation[i])

A = np.empty((17, 17), dtype=object)
B = np.empty((17, 17), dtype=object)
RA = np.empty((17, 17), dtype=object)
RB = np.empty((17, 17), dtype=object)
tA = np.empty((17, 17), dtype=object)
tB = np.empty((17, 17), dtype=object)
K = np.empty((17, 17), dtype=object)
a = np.empty((17, 17), dtype=object)
b = np.empty((17, 17), dtype=object)

I=np.eye(3)
for i in range(0,16):
    for j in range(i+1, 17):
        A[i][j] =(np.linalg.inv(DefaultToBASE[j]))@ DefaultToBASE[i]
        B[i][j] =MarketToCamera[j] @ (np.linalg.inv(MarketToCamera[i]))
        A[i][j] = np.array(A[i][j])
        B[i][j] = np.array(B[i][j])
        RA[i][j] = A[i][j][:3, :3]
        RB[i][j] = B[i][j][:3, :3]
        tA[i][j] = A[i][j][0:3, 3:4]
        tB[i][j] = B[i][j][0:3, 3:4]
        K[i][j] = np.kron(I, RA[i][j]) - np.kron(np.transpose(RB[i][j]), I)
        a[i][j] = RA[i][j] - I

KSUM=K[0][1]
for i in range(0,16):
    for j in range(i+1, 17):
        KSUM = np.vstack((KSUM, K[i][j]))

U, s, Vt = np.linalg.svd(KSUM)
RX_null_space_basis = Vt[-1, :]
RX =np.transpose(RX_null_space_basis.reshape(3, 3))
UX, sX, VtX = np.linalg.svd(RX)
RX = UX @ np.transpose(VtX)
determinant = np.linalg.det(RX)
if determinant < 0:
    RX = -1 * RX
print('RX',RX)

for i in range(0,16):
    for j in range(i+1, 17):
        b[i][j]= RX @ tB[i][j] - tA[i][j]
bSUM = b[0][1]
aSUM = a[0][1]

for i in range(0,16):
    for j in range(i+1, 17):
        bSUM = np.vstack((bSUM, b[i][j]))
        aSUM = np.vstack((aSUM, a[i][j]))

def leastSquares(X, Y):
    result = np.linalg.lstsq(X, Y, rcond=None)
    W = result[0].flatten()
    RSS = result[1][0]
    return W

W = leastSquares(aSUM, bSUM)
print("位移向量:", W)

