import numpy as np
from typing import List

from surfaces.shape import Shape


class Ray:
    def __init__(self, x, y, direction, start_position):
        self.ref_obj = None
        self.sub = None  # secondary ray if glass
        self.hit_coords = []  # on scene hit
        self.pixel_coords = (x, y)  # image pixel location
        self.direction = direction  # ray direction
        self.pos = start_position  # rays start location

    def shoot(self, objects: List[Shape]):
        """
        Shoots the ray and returns the hit object and the hit position
        :param objects: All relevant objects that the ray may hit
        :return: if the ray hit an object, tuple of hit object and the hit position. Otherwise, returns None
        """
        options = {}
        for obj in objects:
            hit, pos = obj.get_intersection_point(self)
            if hit:
                distance = np.linalg.norm(self.pos - pos)
                # Assuming that each hit object has another distance - maybe should change in the future
                options[distance] = (obj, pos)
        if 0 == len(options.keys()):
            return None
        return options[min(options.keys())]

    def compute_triangle(self):
        pass

    def compute_sphere(self):
        pass

    def compute_plane(self):
        pass

    def glass_hit(self):
        pass
