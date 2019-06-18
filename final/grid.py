import cv2
import os
import numpy as np
from cell import Cell

class Grid():
    def __init__(self, img, grid_height, grid_width):
        self.img = cv2.imread(img)                           # original image
        self.img = cv2.resize(self.img, (640, 360))
        self.rows = self.img.shape[0]                        # image height
        self.cols = self.img.shape[1]                        # image width
        self.g_height = self.get_grid_height(grid_height)    # grid height
        self.g_width  = self.get_grid_width(grid_width)      # grid width
        self.cell_height = self.rows // self.g_height        # cell height
        self.cell_width  = self.cols // self.g_width         # cell width

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
        self.warpped_mesh = np.array(self.mesh)

        print ("image size:", self.rows, self.cols)
        print ("grid  size:", self.g_height, self.g_width)
        print ("cell  size:", self.cell_height, self.cell_width)

    def count(self):
        return self.g_width*self.g_height

    def FeatToCellCoor(self, pixel):
        return (pixel[0]//self.cell_height, pixel[1]//self.cell_width)

    def compute_salience(self):
        # the L2 norm of the color varinace inside a cell
        for row in range(self.g_height):
            for col in range(self.g_width):
                row_min, row_end, col_min, col_end = self.gridCell[row][col].get_boundary()
                b_var = np.var(self.img[row_min:row_end, col_min:col_end, 0])
                g_var = np.var(self.img[row_min:row_end, col_min:col_end, 1])
                r_var = np.var(self.img[row_min:row_end, col_min:col_end, 2])
                self.gridCell[row][col].set_salience(np.linalg.norm([b_var, g_var, r_var]) + 0.5)

    def map_texture(self, image):
        self.result_img = np.zeros_like(self.img)
        '''
        for row in range(self.g_height+1):
            for i in range(self.g_width):
                cv2.line(self.result_img,
                         (self.warpped_mesh[row][i][1], self.warpped_mesh[row][i][0]),
                         (self.warpped_mesh[row][i+1][1], self.warpped_mesh[row][i+1][0]), (0,255,0), 2, 1)
        for col in range(self.g_width+1):
            for i in range(self.g_height):
                cv2.line(self.result_img,
                         (self.warpped_mesh[i][col][1], self.warpped_mesh[i][col][0]),
                         (self.warpped_mesh[i+1][col][1], self.warpped_mesh[i+1][col][0]), (0,255,0), 2, 1)
        '''
        print ('computing transform coefficients and mapping texture')
        for cell_row in range(self.g_height):
            for cell_col in range(self.g_width):
                pixel_info = self.gridCell[cell_row][cell_col].compute_pixel_transform_coeff()
                vertices = np.array(list(map(np.array, self.gridCell[cell_row][cell_col].original_v)))
                for pos, coeff in pixel_info:
                    coeff = np.array(coeff)
                    coeff.resize((1, 4))
                    oldPos = np.dot(coeff, vertices).reshape(-1)
                    if oldPos[0] == self.rows: oldPos[0] -= 1
                    if oldPos[1] == self.cols: oldPos[1] -= 1
                    self.result_img[pos[0]][pos[1]] = self.img[int(oldPos[0])][int(oldPos[1])]
                # cv2.imshow(str((cell_row, cell_col)), self.result_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        cv2.imwrite(os.path.join('warpped', image), self.result_img)

    def show_grid(self, name='Grid', feature=None, show=True, save=True, image=None):
        # draw horizontal line
        black = np.zeros_like(self.img)
        for row in range(self.g_height+1):
            for i in range(self.g_width):
                cv2.line(black,
                         (self.mesh[row][i][1], self.mesh[row][i][0]),
                         (self.mesh[row][i+1][1], self.mesh[row][i+1][0]), (255,255,255), 2, 1)
                cv2.line(black,
                         (self.warpped_mesh[row][i][1], self.warpped_mesh[row][i][0]),
                         (self.warpped_mesh[row][i+1][1], self.warpped_mesh[row][i+1][0]), (0,255,0), 2, 1)

        for col in range(self.g_width+1):
            for i in range(self.g_height):
                cv2.line(black,
                         (self.mesh[i][col][1], self.mesh[i][col][0]),
                         (self.mesh[i+1][col][1], self.mesh[i+1][col][0]), (255,255,255), 2, 1)
                cv2.line(black,
                         (self.warpped_mesh[i][col][1], self.warpped_mesh[i][col][0]),
                         (self.warpped_mesh[i+1][col][1], self.warpped_mesh[i+1][col][0]), (0,255,0), 2, 1)

        if feature is not None:
            for feat in feature:
                cv2.drawMarker(black, (feat.col, feat.row), (255,255,255), cv2.MARKER_CROSS, 5, 2)
                cv2.drawMarker(black, (feat.dest_col, feat.dest_row), (0,255,0), cv2.MARKER_CROSS, 2, 1)
                cv2.arrowedLine(black, (feat.col, feat.row), (feat.dest_col, feat.dest_row), (0,0,255), 2, tipLength=0.5)

        if show:
            cv2.namedWindow(name, flags=cv2.WINDOW_NORMAL)
            cv2.imshow(name, black)
            print ('press any key to close window')
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save: cv2.imwrite(os.path.join('grid', image), black)

    def get_grid_height(self, grid_height):
        return self.closest_factor(self.rows, grid_height)

    def get_grid_width(self, grid_width):
        return self.closest_factor(self.cols, grid_width)

    def closest_factor(self, ref, target):
        offset = 0
        while True:
            if ref % (target + offset) == 0: return target + offset
            offset += 1
