from grid import Grid
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
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                if mask[row][col] == 1:
                    self.v_map[len(self.v_map)] = (row, col)
                    self.mesh_map[(row, col)] = len(self.mesh_map)

        # build A
        A = np.zeros((self.feat.size(), len(self.v_map)))
        for i, feat_info in enumerate(self.feat.feat):
            cell_row, cell_col = feat_info[2]
            A[i][self.mesh_map[(cell_row  , cell_col  )]] = feat_info[1][0] # V1
            A[i][self.mesh_map[(cell_row+1, cell_col  )]] = feat_info[1][1] # V2
            A[i][self.mesh_map[(cell_row+1, cell_col+1)]] = feat_info[1][2] # V3
            A[i][self.mesh_map[(cell_row  , cell_col+1)]] = feat_info[1][3] # V4

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
