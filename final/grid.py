import cv2
import numpy as np
from feature import Feature
from cell import Cell

class Grid():
    def __init__(self, img, grid_height, grid_width):
        self.img = cv2.imread(img)                           # original image
        self.rows = self.img.shape[0]                        # image height
        self.cols = self.img.shape[1]                        # image width
        self.g_height = self.get_grid_height(grid_height)    # grid height
        self.g_width  = self.get_grid_width(grid_width)      # grid width
        self.cell_height = self.rows // self.g_height        # cell height
        self.cell_width  = self.cols // self.g_width         # cell width
        self.feat        = Feature()                         # feature object

        # create gridCell map, record by cell
        self.gridCell = [[None for i in range(self.g_width)] for j in range(self.g_height)]
        for row in range(self.g_height):
            for col in range(self.g_width):
                self.gridCell[row][col] = Cell((row*self.cell_height, col*self.cell_width),
                                                self.cell_height,
                                                self.cell_width)

        # create mesh, record by points
        # self.gridCell[row][col] corresponds to four points in thie mesh
        # self.mesh[row][col], self.mesh[row+1][col], self.mesh[row][col+1], self.mesh[row+1][col+1]
        self.mesh = [[None for i in range(self.g_width+1)] for j in range(self.g_height+1)]
        for row in range(self.g_height+1):
            for col in range(self.g_width+1):
                self.mesh[row][col] = np.array([row*self.cell_height, col*self.cell_width])
        self.mesh = np.array(self.mesh)

        print ("image size:", self.rows, self.cols)
        print ("grid  size:", self.g_height, self.g_width)
        print ("cell  size:", self.cell_height, self.cell_width)

    def query_cell(self, pixel):
        return self.gridCell[pixel[0]//self.cell_height][pixel[1]//self.cell_width]

    def read_feature_points(self, filename):
        self.feat.read(filename)

    def compute_bilinear_interpolation(self):
        for feat_name, feat_info in self.feat.feat.items():
            corresponding_cell = self.query_cell(feat_info[0])
            self.feat.set_coefficients(feat_name, corresponding_cell.compute_coeff(feat_info[0]))

    def compute_salience(self):
        # the L2 norm of the color varinace inside a cell
        for row in range(self.g_height):
            for col in range(self.g_width):
                row_min, row_end, col_min, col_end = self.gridCell[row][col].get_boundary()
                b_var = np.var(self.img[row_min:row_end, col_min:col_end, 0])
                g_var = np.var(self.img[row_min:row_end, col_min:col_end, 1])
                r_var = np.var(self.img[row_min:row_end, col_min:col_end, 2])
                self.gridCell[row][col].set_salience(np.linalg.norm([b_var, g_var, r_var]) + 0.5)

    def show_feat(self):
        print (self.feat)

    def show_grid(self):
        # draw horizontal line
        for row in range(self.g_height):
            for i in range(self.g_width):
                cv2.line(self.img,
                         (self.mesh[row][i][1], self.mesh[row][i][0]),
                         (self.mesh[row][i+1][1], self.mesh[row][i+1][0]), (255,255,255), 5, 5)
        for col in range(self.g_width):
            for i in range(self.g_height):
                cv2.line(self.img,
                         (self.mesh[i][col][1], self.mesh[i][col][0]),
                         (self.mesh[i+1][col][1], self.mesh[i+1][col][0]), (255,255,255), 5, 5)
                cv2.circle(self.img,
                           (self.mesh[i][col][1], self.mesh[i][col][0]),
                           5, (0,0,255), thickness=-1)
                cv2.circle(self.img,
                           (self.mesh[i+1][col][1], self.mesh[i+1][col][0]),
                           5, (0,0,255), thickness=-1)

        cv2.namedWindow('Grid', flags=cv2.WINDOW_NORMAL)
        cv2.imshow('Grid', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_grid_height(self, grid_height):
        return self.closest_factor(self.rows, grid_height)

    def get_grid_width(self, grid_width):
        return self.closest_factor(self.cols, grid_width)

    def closest_factor(self, ref, target):
        offset = 0
        while True:
            if ref % (target + offset) == 0: return target + offset
            offset += 1
