from grid import Grid
from feature import Feature

class ContentWarp():
    def __init__(self, image, feat_file, grid_height, grid_width):
        self.feat = Feature() # feature object
        self.read_feature_points(feat_file)

        self.grid = Grid(image, grid_height, grid_width)
        self.grid.compute_salience()
        self.grid.show_grid()

        self.compute_bilinear_interpolation()
        self.build_linear_system()

    def build_linear_system(self):
        pass

    def compute_bilinear_interpolation(self):
        for feat_name, feat_info in self.feat.feat.items():
            corresponding_cell = self.grid.query_cell(feat_info[0])
            self.feat.set_coefficients(feat_name, corresponding_cell.compute_coeff(feat_info[0]))

    def read_feature_points(self, filename):
        self.feat.read(filename)

