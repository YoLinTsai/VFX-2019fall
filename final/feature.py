
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
                line = line.strip().split()
                self.feat.append([(int(line[1]), int(line[2])), None]) #[(row, col), bilinear interpolation coefficients]

    def set_coefficients(self, i, coeff):
        self.feat[i][1] = coeff

    def size(self):
        return len(self.feat)

    def __str__(self):
        message = ""
        for feat_info in self.feat:
            message += "row:{:4d}".format(feat_info[0][0])
            message += " col:{:4d} ".format(feat_info[0][1])
            message += str(feat_info[1]) + '\n'
        return message
