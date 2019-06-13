from grid import Grid
from cell import R90
from feature import Feature
import numpy as np

class ContentWarp():
    def __init__(self, image, feat_file, grid_height, grid_width):
        self.feat = Feature() # feature object
        self.read_feature_points(feat_file)

        self.grid = Grid(image, grid_height, grid_width)
        self.grid.compute_salience()
        # self.grid.show_grid()

        self.set_grid_info_to_feat()
        self.compute_bilinear_interpolation()
        self.build_linear_system()

    def build_linear_system(self):
        '''
        A: [w1 0 w2 0 w3 0 w4 0 0 0 ... 0] X: [V_1x]
           [0 w1 0 w2 0 w3 0 w4 0 0 ... 0]    [V_1y]
                          .                   [V_2x]
                          .                   [V_2y]
                          .                   [V_3x]
               simularity transform           [V_3y]
                          .                   [V_4x]
                          .                   [V_4y]
                          .                     .
                          .                     .
                                                .
        '''
        # A*x = B
        self.v_map    = dict() # the map from Xi to mesh coordinates
        self.mesh_map = dict() # the map from mesh coordinates to Xi

        # find the vertices that need to be adjusted
        mask = np.zeros_like(self.grid.mesh[:,:,0])
        for feat_info in self.feat.feat:
            cell_row, cell_col = feat_info[2]
            mask[cell_row  ][cell_col  ] = 1
            mask[cell_row  ][cell_col+1] = 1
            mask[cell_row+1][cell_col  ] = 1
            mask[cell_row+1][cell_col+1] = 1

        # collect these vertices (variables)
        map_id = 0 # if x[i] x[i+1] would be the row and col respectively for every even i
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                if mask[row][col] == 1:
                    print (row, col)
                    self.v_map[map_id] = (row, col)
                    self.mesh_map[(row, col)] = map_id
                    map_id += 2

        # build A_DataTerm and B
        # build A_SimularityTransform
        # temporarily set the new feature points to row, col
        check_grid   = set() # check if a grid is visited more than once for simularity transform term
        A_simularity = np.zeros((1, 2*len(self.v_map)))
        B_simularity = np.zeros((1, 1))
        A_data = np.zeros((2*self.feat.size(), 2*len(self.v_map)))
        B_data = np.zeros((2*self.feat.size(), 1))
        for i, feat_info in enumerate(self.feat.feat):
            cell_row, cell_col = feat_info[2]

            # data term
            v1_x_pos = self.mesh_map[(cell_row  , cell_col  )]; v1_y_pos = v1_x_pos + 1
            v2_x_pos = self.mesh_map[(cell_row+1, cell_col  )]; v2_y_pos = v2_x_pos + 1
            v3_x_pos = self.mesh_map[(cell_row+1, cell_col+1)]; v3_y_pos = v3_x_pos + 1
            v4_x_pos = self.mesh_map[(cell_row  , cell_col+1)]; v4_y_pos = v4_x_pos + 1

            A_data[2*i][v1_x_pos] = feat_info[1][0] # V1's coeff for x coordinate
            A_data[2*i][v2_x_pos] = feat_info[1][1] # V2's coeff for x coordinate
            A_data[2*i][v3_x_pos] = feat_info[1][2] # V3's coeff for x coordinate
            A_data[2*i][v4_x_pos] = feat_info[1][3] # V4's coeff for x coordinate

            A_data[2*i+1][v1_y_pos] = feat_info[1][0] # V1's coeff for y coordinate
            A_data[2*i+1][v2_y_pos] = feat_info[1][1] # V2's coeff for y coordinate
            A_data[2*i+1][v3_y_pos] = feat_info[1][2] # V3's coeff for y coordinate
            A_data[2*i+1][v4_y_pos] = feat_info[1][3] # V4's coeff for y coordinate

            B_data[2*i]   = np.array(feat_info[0][0]) + 0.5 # to grid coordinate
            B_data[2*i+1] = np.array(feat_info[0][1]) + 0.5 # to grid coordinate

            # simularity transfrom term
            if (cell_row, cell_col) not in check_grid:
                Ws = self.grid.gridCell[cell_row][cell_col].salience
                A_Sim_new = np.zeros((16, 2*len(self.v_map)))
                B_Sim_new = np.zeros((16, 1))

                # first triangle
                u  = self.grid.gridCell[cell_row][cell_col].u_v[0][0]
                v  = self.grid.gridCell[cell_row][cell_col].u_v[0][1]
                A_Sim_new[0][v1_x_pos] = Ws * ( 1 )
                A_Sim_new[0][v2_x_pos] = Ws * ( u - 1 )
                A_Sim_new[0][v2_y_pos] = Ws * ( v )
                A_Sim_new[0][v3_x_pos] = Ws * ( -u )
                A_Sim_new[0][v3_y_pos] = Ws * ( -v )
                B_Sim_new[0] = 0
                A_Sim_new[1][v1_y_pos] = Ws * ( 1 )
                A_Sim_new[1][v2_y_pos] = Ws * ( u - 1 )
                A_Sim_new[1][v2_x_pos] = Ws * ( -v )
                A_Sim_new[1][v3_y_pos] = Ws * ( -u )
                A_Sim_new[1][v3_x_pos] = Ws * ( v )
                B_Sim_new[1] = 0

                # second triangle
                u  = self.grid.gridCell[cell_row][cell_col].u_v[1][0]
                v  = self.grid.gridCell[cell_row][cell_col].u_v[1][1]
                A_Sim_new[2][v1_x_pos] = Ws * ( 1 )
                A_Sim_new[2][v4_x_pos] = Ws * ( u - 1 )
                A_Sim_new[2][v4_y_pos] = Ws * ( v )
                A_Sim_new[2][v3_x_pos] = Ws * ( -u )
                A_Sim_new[2][v3_y_pos] = Ws * ( -v )
                B_Sim_new[2] = 0
                A_Sim_new[3][v1_y_pos] = Ws * ( 1 )
                A_Sim_new[3][v4_y_pos] = Ws * ( u - 1 )
                A_Sim_new[3][v4_x_pos] = Ws * ( -v )
                A_Sim_new[3][v3_y_pos] = Ws * ( -u )
                A_Sim_new[3][v3_x_pos] = Ws * ( v )
                B_Sim_new[3] = 0

                # third triangle
                u  = self.grid.gridCell[cell_row][cell_col].u_v[2][0]
                v  = self.grid.gridCell[cell_row][cell_col].u_v[2][1]
                A_Sim_new[4][v2_x_pos] = Ws * ( 1 )
                A_Sim_new[4][v3_x_pos] = Ws * ( u - 1 )
                A_Sim_new[4][v3_y_pos] = Ws * ( v )
                A_Sim_new[4][v4_x_pos] = Ws * ( -u )
                A_Sim_new[4][v4_y_pos] = Ws * ( -v )
                B_Sim_new[4] = 0
                A_Sim_new[5][v2_y_pos] = Ws * ( 1 )
                A_Sim_new[5][v3_y_pos] = Ws * ( u - 1 )
                A_Sim_new[5][v3_x_pos] = Ws * ( -v )
                A_Sim_new[5][v4_y_pos] = Ws * ( -u )
                A_Sim_new[5][v4_x_pos] = Ws * ( v )
                B_Sim_new[5] = 0

                # forth triangle
                u  = self.grid.gridCell[cell_row][cell_col].u_v[3][0]
                v  = self.grid.gridCell[cell_row][cell_col].u_v[3][1]
                A_Sim_new[6][v2_x_pos] = Ws * ( 1 )
                A_Sim_new[6][v1_x_pos] = Ws * ( u - 1 )
                A_Sim_new[6][v1_y_pos] = Ws * ( v )
                A_Sim_new[6][v4_x_pos] = Ws * ( -u )
                A_Sim_new[6][v4_y_pos] = Ws * ( -v )
                B_Sim_new[6] = 0
                A_Sim_new[7][v2_y_pos] = Ws * ( 1 )
                A_Sim_new[7][v1_y_pos] = Ws * ( u - 1 )
                A_Sim_new[7][v1_x_pos] = Ws * ( -v )
                A_Sim_new[7][v4_y_pos] = Ws * ( -u )
                A_Sim_new[7][v4_x_pos] = Ws * ( v )
                B_Sim_new[7] = 0

                # fifth triangle
                u  = self.grid.gridCell[cell_row][cell_col].u_v[4][0]
                v  = self.grid.gridCell[cell_row][cell_col].u_v[4][1]
                A_Sim_new[8][v3_x_pos] = Ws * ( 1 )
                A_Sim_new[8][v4_x_pos] = Ws * ( u - 1 )
                A_Sim_new[8][v4_y_pos] = Ws * ( v )
                A_Sim_new[8][v1_x_pos] = Ws * ( -u )
                A_Sim_new[8][v1_y_pos] = Ws * ( -v )
                B_Sim_new[8] = 0
                A_Sim_new[9][v3_y_pos] = Ws * ( 1 )
                A_Sim_new[9][v4_y_pos] = Ws * ( u - 1 )
                A_Sim_new[9][v4_x_pos] = Ws * ( -v )
                A_Sim_new[9][v1_y_pos] = Ws * ( -u )
                A_Sim_new[9][v1_x_pos] = Ws * ( v )
                B_Sim_new[9] = 0

                # sixth triangle
                u  = self.grid.gridCell[cell_row][cell_col].u_v[5][0]
                v  = self.grid.gridCell[cell_row][cell_col].u_v[5][1]
                A_Sim_new[10][v3_x_pos] = Ws * ( 1 )
                A_Sim_new[10][v2_x_pos] = Ws * ( u - 1 )
                A_Sim_new[10][v2_y_pos] = Ws * ( v )
                A_Sim_new[10][v1_x_pos] = Ws * ( -u )
                A_Sim_new[10][v1_y_pos] = Ws * ( -v )
                B_Sim_new[10] = 0
                A_Sim_new[11][v3_y_pos] = Ws * ( 1 )
                A_Sim_new[11][v2_y_pos] = Ws * ( u - 1 )
                A_Sim_new[11][v2_x_pos] = Ws * ( -v )
                A_Sim_new[11][v1_y_pos] = Ws * ( -u )
                A_Sim_new[11][v1_x_pos] = Ws * ( v )
                B_Sim_new[11] = 0

                # seventh triangle
                u  = self.grid.gridCell[cell_row][cell_col].u_v[6][0]
                v  = self.grid.gridCell[cell_row][cell_col].u_v[6][1]
                A_Sim_new[12][v4_x_pos] = Ws * ( 1 )
                A_Sim_new[12][v1_x_pos] = Ws * ( u - 1 )
                A_Sim_new[12][v1_y_pos] = Ws * ( v )
                A_Sim_new[12][v2_x_pos] = Ws * ( -u )
                A_Sim_new[12][v2_y_pos] = Ws * ( -v )
                B_Sim_new[12] = 0
                A_Sim_new[13][v4_y_pos] = Ws * ( 1 )
                A_Sim_new[13][v1_y_pos] = Ws * ( u - 1 )
                A_Sim_new[13][v1_x_pos] = Ws * ( -v )
                A_Sim_new[13][v2_y_pos] = Ws * ( -u )
                A_Sim_new[13][v2_x_pos] = Ws * ( v )
                B_Sim_new[13] = 0

                # eighth triangle
                u  = self.grid.gridCell[cell_row][cell_col].u_v[7][0]
                v  = self.grid.gridCell[cell_row][cell_col].u_v[7][1]
                A_Sim_new[14][v4_x_pos] = Ws * ( 1 )
                A_Sim_new[14][v3_x_pos] = Ws * ( u - 1 )
                A_Sim_new[14][v3_y_pos] = Ws * ( v )
                A_Sim_new[14][v2_x_pos] = Ws * ( -u )
                A_Sim_new[14][v2_y_pos] = Ws * ( -v )
                B_Sim_new[14] = 0
                A_Sim_new[15][v4_y_pos] = Ws * ( 1 )
                A_Sim_new[15][v3_y_pos] = Ws * ( u - 1 )
                A_Sim_new[15][v3_x_pos] = Ws * ( -v )
                A_Sim_new[15][v2_y_pos] = Ws * ( -u )
                A_Sim_new[15][v2_x_pos] = Ws * ( v )
                B_Sim_new[15] = 0

                A_simularity = np.vstack((A_simularity, A_Sim_new))
                B_simularity = np.vstack((B_simularity, B_Sim_new))

                check_grid.add((cell_row, cell_col))

        A = np.vstack((A_data, A_simularity[1:]))
        B = np.vstack((B_data, B_simularity[1:]))

        print (A.shape)
        A = np.vstack((A, np.array([1,0,0,0,0,0,0,0,0,0,0,0])))
        A = np.vstack((A, np.array([0,1,0,0,0,0,0,0,0,0,0,0])))
        B = np.vstack((B, np.array([0])))
        B = np.vstack((B, np.array([0])))

        print (A)
        print (B)

        rank_A = np.linalg.matrix_rank(A)
        if rank_A < A.shape[1]:
            print ('linear system is underdetermined!')
            print ('Solution is not unique!')
        elif rank_A == A.shape[1]:
            print ('Solution is unique.')
        else:
            print ('linear system is overdetermined!')
            print ('Calculating least square solution.')

        X, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

        # round the solution to the second decimal
        X = np.array([ round(x, 2) for x in X.reshape(-1) ]).reshape((-1, 1))
        print (X)

    def compute_bilinear_interpolation(self):
        for i, feat_info in enumerate(self.feat.feat):
            corresponding_cell = self.grid.gridCell[feat_info[2][0]][feat_info[2][1]]
            self.feat.set_coefficients(i, corresponding_cell.compute_coeff(feat_info[0]))
        print (self.feat)

    def read_feature_points(self, filename):
        self.feat.read(filename)

    def set_grid_info_to_feat(self):
        for i, feat_info in enumerate(self.feat.feat):
            self.feat.set_grid_position(i, self.grid.FeatToCellCoor(feat_info[0]))
