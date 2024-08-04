import numpy as np
from surfaces.shape import Shape


class Sphere(Shape):
    def get_factor(self):
        return self.radius

    def __init__(self, position, radius, material_index):
        super(Sphere).__init__()
        self.position = position
        self.radius = radius
        self.material_index = material_index

    def get_intersection_point(self, start_point,direction) -> (bool, np.array):
        # According to lecture 7, page 39
        a = 1
        b = 2 * np.dot(direction, (start_point - self.position)) #TODO add max in the dot?
        c = np.linalg.norm(start_point - self.position) ** 2 - self.radius ** 2 #TODO why self.position as C?
        det = b ** 2 - 4 * a * c
        if det < 0:
            return False, np.array([0, 0, 0])
        t1 = (-b + det ** 0.5) / (2 * a)
        t2 = (-b - det ** 0.5) / (2 * a)
        t = min(t1, t2)
        if t < 0:
            return False, np.array([0, 0, 0])
        return True, start_point + direction * t
