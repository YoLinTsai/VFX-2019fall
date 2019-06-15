import numpy as np

'''
feature container
'''


class Feature():
    def __init__(self, filename=None):
        if filename is not None: self.read(filename)
        self.feat = list()

    def read(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('// '): continue
                line = line.strip().split()

                #[row, col, temporal coherence coeff, bilinear interpolation coefficients, cooresponding gridCell position]
                self.feat.append(FeatInfo(int(line[1]), int(line[2]), float(line[3]), None, None))

    def add_noise(self):
        for i in range(len(self.feat)):
            self.feat[i].set_dest(self.feat[i].row + round(5*np.random.randn()), self.feat[i].col + round(5*np.random.randn()))

    def set_coefficients(self, i, coeff):
        self.feat[i].interpolation_coeff = coeff

    def set_grid_position(self, i, pos):
        self.feat[i].grid_pos = pos

    def size(self):
        return len(self.feat)

    def __str__(self):
        message = ""
        for feat_info in self.feat:
            message += "row:{:4d}".format(feat_info.row)
            message += " col:{:4d} ".format(feat_info.col)
            message += " coeff: " + str(feat_info.interpolation_coeff)
            message += " gridCell: " + str(feat_info.grid_pos) + '\n'
        return message

class FeatInfo():
    def __init__(self, row, col, tc, bic, gp):
        self.row = row
        self.col = col
        self.pos = (row, col)
        self.temporal_coeff = tc
        self.interpolation_coeff = bic
        self.grid_pos = gp
        self.dest_row = None
        self.dest_row = None

    def set_dest(self, row, col):
        self.dest_row = row
        self.dest_col = col
