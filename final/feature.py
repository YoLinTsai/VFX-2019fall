import numpy as np

'''
feature container
'''


class Feature():
    def __init__(self, filename=None):
        if filename is not None: self.read(filename)
        self.feat = list()

    def read(self, filename, margin):
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('// '): continue
                line = line.strip().split()

                #[row, col, temporal coherence coeff, bilinear interpolation coefficients, cooresponding gridCell position]
                self.feat.append(FeatInfo(round(float(line[1]))+margin, round(float(line[2]))+margin, float(line[3]), None, None))
                self.feat[-1].set_dest(round(float(line[4]))+margin, round(float(line[5]))+margin)
                # if self.feat[-1].illegal():
                    # print ('removing feature ->', self.feat[-1])
                    # del self.feat[-1]

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
            message += " d_row:{:4d}".format(feat_info.dest_row)
            message += " d_col:{:4d} ".format(feat_info.dest_col)
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

    def set_global(self, row, col):
        self.global_row = row
        self.global_col = col
        self.global_pos = (row, col)

    def illegal(self):
        if self.dest_row >= 720+200: return True
        if self.dest_row < 200: return True
        if self.dest_col >= 1280+200: return True
        if self.dest_col < 200: return True

    def __str__(self):
        message = ""
        message += "row:{:4d}".format(self.row)
        message += " col:{:4d} ".format(self.col)
        message += " d_row:{:4d}".format(self.dest_row)
        message += " d_col:{:4d} ".format(self.dest_col)
        message += " coeff: " + str(self.interpolation_coeff)
        message += " gridCell: " + str(self.grid_pos)
        return message
