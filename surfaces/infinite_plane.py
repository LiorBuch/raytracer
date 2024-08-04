import numpy as np
from surfaces.shape import Shape


class InfinitePlane(Shape):
    def get_factor(self):
        pass

    def __init__(self, normal, offset, material_index):
        super(InfinitePlane).__init__()
        if np.sum(normal == [0, 0, 0]) == 3:
            raise Exception("Invalid plane!")
        self.normal = normal
        self.offset = offset
        self.material_index = material_index

    def get_intersection_point(self, start_point,direction) -> (bool, np.array):

        if np.dot(direction, self.normal) == 0:
            return False, np.array([0, 0, 0])
        # Since the formula of the plane is P * N = c, not P * N + c = 0
        t = -(np.dot(start_point, self.normal) - self.offset) / (np.dot(direction, self.normal)) #TODO add max in the dot?
        if t < 0:
            return False, np.array([0, 0, 0])
        return True, start_point + t * direction

    def get_point_on_plane(self):
        x,y,z = (1,1,1)
        if self.normal[0]!=0:
            x = (self.offset - np.dot(self.normal[1], y) - np.dot(self.normal[2], z)) / self.normal[0]
        elif self.normal[1]!=0:
            y = (self.offset - np.dot(self.normal[0], x) - np.dot(self.normal[2], z)) / self.normal[1]
        else:
            z = (self.offset - np.dot(self.normal[0], x) - np.dot(self.normal[1], y)) / self.normal[0]
        return np.array([x,y,z])
