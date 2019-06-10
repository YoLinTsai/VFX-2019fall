
'''
feature container
'''


class Feature():
    def __init__(self, filename=None):
        if filename is not None: self.read(filename)
        self.feat = dict()

    def read(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip().split()
                self.feat[line[0]] = [(int(line[1]), int(line[2])), None] #[(row, col), bilinear interpolation coefficients]

    def set_coefficients(self, feat_name, coeff):
        self.feat[feat_name][1] = coeff

    def __str__(self):
        message = ""
        for feat_name, feat_info in self.feat.items():
            message += feat_name + " row:{:4d}".format(feat_info[0][0])
            message += " col:{:4d} ".format(feat_info[0][1])
            message += str(feat_info[1]) + '\n'
        return message
