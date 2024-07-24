import numpy as np
from ray import Ray
from surfaces.shape import Shape


class Cube(Shape):
    def get_factor(self):
        return self.scale

    def __init__(self, position, scale, material_index):
        self.position = position
        self.scale = scale/2.0
        self.material_index = material_index

    def get_intersection_point(self, ray: Ray) -> (bool, np.array):
        start_point = ray.pos
        direction = ray.direction
        # According to ChatGPT
        half_scale = self.scale / 2.0

        min_bound = self.position - half_scale
        max_bound = self.position + half_scale

        t_min = (min_bound - start_point) / direction
        t_max = (max_bound - start_point) / direction

        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)

        t_near = np.max(t1)
        t_far = np.min(t2)

        if t_near <= t_far and t_far >= 0:
            return True, np.array(start_point + t_near * direction)
        return False, np.array([0, 0, 0])
