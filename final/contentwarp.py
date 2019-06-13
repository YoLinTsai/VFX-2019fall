from grid import Grid
from cell import R90
from feature import Feature
import numpy as np

class ContentWarp():
    def __init__(self, image, feat_file, grid_height, grid_width, alpha=1):
        self.alpha = alpha
        self.feat = Feature() # feature object
        self.read_feature_points(feat_file)

        self.grid = Grid(image, grid_height, grid_width)
        self.grid.compute_salience()
        # self.grid.show_grid()

        self.set_grid_info_to_feat()
        self.compute_bilinear_interpolation()
        self.build_linear_system_and_solve()
        self.grid.compute_cell_pixels()
        self.grid.show_grid()

    def build_linear_system_and_solve(self):
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
        v_map    = dict() # the map from Xi to mesh coordinates
        mesh_map = dict() # the map from mesh coordinates to Xi

        # find the vertices that need to be adjusted
        mask = np.zeros_like(self.grid.mesh[:,:,0])
        for feat_info in self.feat.feat:
            cell_row, cell_col = feat_info.grid_pos
            mask[cell_row  ][cell_col  ] = 1
            mask[cell_row  ][cell_col+1] = 1
            mask[cell_row+1][cell_col  ] = 1
            mask[cell_row+1][cell_col+1] = 1

        # collect these vertices (variables)
        map_id = 0 # if x[i] x[i+1] would be the row and col respectively for every even i
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                if mask[row][col] == 1:
                    v_map[map_id] = (row, col)
                    mesh_map[(row, col)] = map_id
                    map_id += 2

        # build A_DataTerm and B
        # build A_SimularityTransform
        # temporarily set the new feature points to row, col
        check_grid   = set() # check if a grid is visited more than once for simularity transform term
        A_simularity = np.zeros((1, 2*len(v_map)))
        B_simularity = np.zeros((1, 1))
        A_data = np.zeros((2*self.feat.size(), 2*len(v_map)))
        B_data = np.zeros((2*self.feat.size(), 1))
        for i, feat_info in enumerate(self.feat.feat):
            cell_row, cell_col = feat_info.grid_pos
            tl = feat_info.temporal_coeff

            # data term
            v1_x_pos = mesh_map[(cell_row  , cell_col  )]; v1_y_pos = v1_x_pos + 1
            v2_x_pos = mesh_map[(cell_row+1, cell_col  )]; v2_y_pos = v2_x_pos + 1
            v3_x_pos = mesh_map[(cell_row+1, cell_col+1)]; v3_y_pos = v3_x_pos + 1
            v4_x_pos = mesh_map[(cell_row  , cell_col+1)]; v4_y_pos = v4_x_pos + 1

            A_data[2*i][v1_x_pos] = tl * feat_info.interpolation_coeff[0] # V1's coeff for x coordinate
            A_data[2*i][v2_x_pos] = tl * feat_info.interpolation_coeff[1] # V2's coeff for x coordinate
            A_data[2*i][v3_x_pos] = tl * feat_info.interpolation_coeff[2] # V3's coeff for x coordinate
            A_data[2*i][v4_x_pos] = tl * feat_info.interpolation_coeff[3] # V4's coeff for x coordinate

            A_data[2*i+1][v1_y_pos] = tl * feat_info.interpolation_coeff[0] # V1's coeff for y coordinate
            A_data[2*i+1][v2_y_pos] = tl * feat_info.interpolation_coeff[1] # V2's coeff for y coordinate
            A_data[2*i+1][v3_y_pos] = tl * feat_info.interpolation_coeff[2] # V3's coeff for y coordinate
            A_data[2*i+1][v4_y_pos] = tl * feat_info.interpolation_coeff[3] # V4's coeff for y coordinate

            B_data[2*i]   = tl * ( np.array(feat_info.row) + 0.5 + 10) # to grid coordinate
            B_data[2*i+1] = tl * ( np.array(feat_info.col) + 0.5 + 10) # to grid coordinate

            # simularity transfrom term
            if (cell_row, cell_col) not in check_grid:
                Ws = self.grid.gridCell[cell_row][cell_col].salience
                A_Sim_new = np.zeros((16, 2*len(v_map)))
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

        A_simularity *= self.alpha
        B_simularity *= self.alpha

        A = np.vstack((A_data, A_simularity[1:]))
        B = np.vstack((B_data, B_simularity[1:]))

        print (A.shape)

        '''
        due to the fact that during testing the system is very likely to be under-determined,
        these two terms are added to set the upper left vertex to (0, 0) in order to verfiy
        the correctness of the implementation
        (a feature that belongs the the grid[0][0] must be specified in feat.txt)
        '''
        A = np.vstack((A, np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0])))
        A = np.vstack((A, np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0])))
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


        # apply the result
        for i in range(X.shape[0]):
            if i % 2 != 0: continue
            mesh_row, mesh_col = v_map[i]
            self.grid.mesh[mesh_row][mesh_col] = np.array([X[i][0], X[i+1][0]])

        for cell_row, cell_col in check_grid:
            v1 = self.grid.mesh[cell_row  ][cell_col  ]
            v2 = self.grid.mesh[cell_row+1][cell_col  ]
            v3 = self.grid.mesh[cell_row+1][cell_col+1]
            v4 = self.grid.mesh[cell_row  ][cell_col+1]
            self.grid.gridCell[cell_row][cell_col].set_corners(v1, v2, v3, v4)

    def compute_bilinear_interpolation(self):
        for i, feat_info in enumerate(self.feat.feat):
            corresponding_cell = self.grid.gridCell[feat_info.grid_pos[0]][feat_info.grid_pos[1]]
            self.feat.set_coefficients(i, corresponding_cell.compute_coeff(feat_info.pos))
        print (self.feat)

    def read_feature_points(self, filename):
        self.feat.read(filename)

    def set_grid_info_to_feat(self):
        for i, feat_info in enumerate(self.feat.feat):
            self.feat.set_grid_position(i, self.grid.FeatToCellCoor(feat_info.pos))
