import numpy as np
from ray import Ray
from surfaces.shape import Shape


class InfinitePlane(Shape):
    def __init__(self, normal, offset, material_index):
        self.normal = normal
        self.offset = offset
        self.material_index = material_index

    def get_intersection_point(self, ray: Ray) -> (bool, np.array):
        start_point = ray.pos
        direction = ray.direction
        t = -(np.dot(start_point, self.normal) + self.offset) / (np.dot(direction, self.normal))
        if t < 0:
            return False, np.array([0, 0, 0])
        return True, start_point + t * direction
