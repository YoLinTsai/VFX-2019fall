import numpy as np


R90 = np.zeros((2, 2))
R90[0][1] = 1
R90[1][0] = -1

class Cell():
    '''
    the pixel coordinates correspond to the upper left of the grid coordinates
    (0, 0)        (0, 2)
    V1＿＿＿＿＿＿V4
    |      |       |
    | (0,0)| (0,1) |
    |＿＿＿|＿＿＿ |
    |      |       |
    | (1,0)| (1,1) |
    |＿＿＿|＿＿＿ |
    V2            V3
    (2, 0)         (2, 2)

    V1 = V2 + u1_1*(V3 - V2) + v1_1*R90*(V3 - V2)        R90: [0,  1]     self.u_v: [u1_1, v1_1]
    V1 = V4 + u1_2*(V3 - V4) + v1_2*R90*(V3 - V4)             [-1, 0]               [u1_2, v1_2]
                                                                                    [u2_1, v2_1]
    V2 = V3 + u2_1*(V4 - V3) + v2_1*R90*(V4 - V3)                                   [u2_2, v2_2]
    V2 = V1 + u2_2*(V4 - V1) + v2_2*R90*(V4 - V1)                                   [u3_1, v3_1]
                                                                                    [u3_2, v3_2]
    V3 = V4 + u3_1*(V1 - V4) + v3_1*R90*(V1 - V4)                                   [u4_1, v4_1]
    V3 = V2 + u3_2*(V1 - V2) + v3_2*R90*(V1 - V2)                                   [u4_2, v4_2]

    V4 = V1 + u4_1*(V2 - V1) + v4_1*R90*(V2 - V1)
    V4 = V3 + u4_2*(V2 - V3) + v4_2*R90*(V2 - V3)

    '''
    def __init__(self, v1, height, width):
        self.v1 = v1
        self.v2 = (v1[0] + height, v1[1])
        self.v3 = (v1[0] + height, v1[1] + width)
        self.v4 = (v1[0],          v1[1] + width)
        self.compute_u_v()

    def compute_coeff(self, pixel):
        x, y = pixel[0]+0.5, pixel[1]+0.5
        cv1 = abs(x - self.v3[0]) * abs(y - self.v3[1])
        cv2 = abs(x - self.v4[0]) * abs(y - self.v4[1])
        cv3 = abs(x - self.v1[0]) * abs(y - self.v1[1])
        cv4 = abs(x - self.v2[0]) * abs(y - self.v2[1])
        return np.array([cv1, cv2, cv3, cv4]) / (cv1 + cv2 + cv3 + cv4)

    def compute_u_v(self):
        self.u_v = np.zeros((8, 2))

        vec_21 = np.array(self.v1) - np.array(self.v2)
        vec_23 = np.array(self.v3) - np.array(self.v2)
        vec_90 = np.dot(R90, vec_23.reshape((2, 1))).reshape(-1)
        self.u_v[0][0] = np.dot(vec_21, vec_23) / (vec_23**2).sum()
        self.u_v[0][1] = np.dot(vec_21, vec_90) / (vec_90**2).sum()

        vec_41 = np.array(self.v1) - np.array(self.v4)
        vec_43 = np.array(self.v3) - np.array(self.v4)
        vec_90 = np.dot(R90, vec_43.reshape((2, 1))).reshape(-1)
        self.u_v[1][0] = np.dot(vec_41, vec_43) / (vec_43**2).sum()
        self.u_v[1][1] = np.dot(vec_41, vec_90) / (vec_90**2).sum()

        vec_32 = np.array(self.v2) - np.array(self.v3)
        vec_34 = np.array(self.v4) - np.array(self.v3)
        vec_90 = np.dot(R90, vec_34.reshape((2, 1))).reshape(-1)
        self.u_v[2][0] = np.dot(vec_32, vec_34) / (vec_34**2).sum()
        self.u_v[2][1] = np.dot(vec_32, vec_90) / (vec_90**2).sum()

        vec_12 = np.array(self.v2) - np.array(self.v1)
        vec_14 = np.array(self.v4) - np.array(self.v1)
        vec_90 = np.dot(R90, vec_14.reshape((2, 1))).reshape(-1)
        self.u_v[3][0] = np.dot(vec_12, vec_14) / (vec_14**2).sum()
        self.u_v[3][1] = np.dot(vec_12, vec_90) / (vec_90**2).sum()

        vec_43 = np.array(self.v3) - np.array(self.v4)
        vec_41 = np.array(self.v1) - np.array(self.v4)
        vec_90 = np.dot(R90, vec_41.reshape((2, 1))).reshape(-1)
        self.u_v[4][0] = np.dot(vec_43, vec_41) / (vec_41**2).sum()
        self.u_v[4][1] = np.dot(vec_43, vec_90) / (vec_90**2).sum()

        vec_23 = np.array(self.v3) - np.array(self.v2)
        vec_21 = np.array(self.v1) - np.array(self.v2)
        vec_90 = np.dot(R90, vec_21.reshape((2, 1))).reshape(-1)
        self.u_v[5][0] = np.dot(vec_23, vec_21) / (vec_21**2).sum()
        self.u_v[5][1] = np.dot(vec_23, vec_90) / (vec_90**2).sum()

        vec_14 = np.array(self.v4) - np.array(self.v1)
        vec_12 = np.array(self.v2) - np.array(self.v1)
        vec_90 = np.dot(R90, vec_12.reshape((2, 1))).reshape(-1)
        self.u_v[6][0] = np.dot(vec_14, vec_12) / (vec_12**2).sum()
        self.u_v[6][1] = np.dot(vec_14, vec_90) / (vec_90**2).sum()

        vec_34 = np.array(self.v4) - np.array(self.v3)
        vec_32 = np.array(self.v2) - np.array(self.v3)
        vec_90 = np.dot(R90, vec_32.reshape((2, 1))).reshape(-1)
        self.u_v[7][0] = np.dot(vec_34, vec_32) / (vec_32**2).sum()
        self.u_v[7][1] = np.dot(vec_34, vec_90) / (vec_90**2).sum()

    def get_boundary(self):
        # returns the boundary in the pixel coordinates
        return self.v1[0], self.v2[0], self.v1[1], self.v4[1]

    def set_salience(self, s):
        self.salience = s

    def __str__(self):
        message = str(self.v1) + str(self.v4) + '\n'
        message += str(self.v2) + str(self.v3) + '\n'
        return message
