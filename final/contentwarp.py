from grid import Grid

class ContentWarp():
    def __init__(self, image, feat_file, grid_height, grid_width):
        self.grid = Grid(image, grid_height, grid_width)
        self.grid.read_feature_points(feat_file)
        self.grid.compute_bilinear_interpolation()
        self.grid.compute_salience()
        self.grid.show_grid()

