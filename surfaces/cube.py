import numpy as np
from surfaces.shape import Shape


class Cube(Shape):
    def get_factor(self):
        return self.scale

    def __init__(self, position, scale, material_index):
        super(Cube).__init__()
        self.position = position
        self.scale = scale/2.0
        self.material_index = material_index

    def get_intersection_point(self, start_point,direction) -> (bool, np.array):
        x_c, y_c, z_c = self.position

        # Calculate the bounds of the cube in each axis
        min_x, max_x = x_c - self.scale, x_c + self.scale
        min_y, max_y = y_c - self.scale, y_c + self.scale
        min_z, max_z = z_c - self.scale, z_c + self.scale

        x, y, z = start_point
        dx, dy, dz = direction

        # If the direction is 0 we can't divide by it
        # So we will put the t_max and t_min according to x relation with the min_x and max_x.
        # Same for y and z
        if dx != 0:
            t_min_x = (min_x - x) / dx
            t_max_x = (max_x - x) / dx
            if t_min_x > t_max_x: t_min_x, t_max_x = t_max_x, t_min_x
        else:
            t_min_x = -np.inf if x < min_x or x > max_x else 0
            t_max_x = np.inf if x >= min_x and x <= max_x else 0

        if dy != 0:
            t_min_y = (min_y - y) / dy
            t_max_y = (max_y - y) / dy
            if t_min_y > t_max_y: t_min_y, t_max_y = t_max_y, t_min_y
        else:
            t_min_y = -np.inf if y < min_y or y > max_y else 0
            t_max_y = np.inf if y >= min_y and y <= max_y else 0

        if dz != 0:
            t_min_z = (min_z - z) / dz
            t_max_z = (max_z - z) / dz
            if t_min_z > t_max_z: t_min_z, t_max_z = t_max_z, t_min_z
        else:
            t_min_z = -np.inf if z < min_z or z > max_z else 0
            t_max_z = np.inf if z >= min_z and z <= max_z else 0

        # Find the largest t_min and the smallest t_max across all slabs
        t_enter = max(t_min_x, t_min_y, t_min_z)
        t_exit = min(t_max_x, t_max_y, t_max_z)

        # Check if there is an intersection
        if t_enter > t_exit or t_exit < 0:
            return False, np.array([0, 0, 0])  # No intersection

        # Calculate the intersection point
        intersection_point = np.array([
            x + t_enter * dx,
            y + t_enter * dy,
            z + t_enter * dz
        ])

        return True, intersection_point


    def get_normal(self, hit_pos):
        x_c, y_c, z_c = self.position  # Center coordinates
        x, y, z = hit_pos

        dx = abs(x - x_c) - self.scale
        dy = abs(y - y_c) - self.scale
        dz = abs(z - z_c) - self.scale
        normal = np.array([0.0, 0.0, 0.0])

        # Find the maximum distance (which slab the hit point is closest to)
        max_dist = max(dx, dy, dz)

        if np.isclose(max_dist, dx):
            normal[0] = np.sign(x - x_c)
        elif np.isclose(max_dist, dy):
            normal[1] = np.sign(y - y_c)
        elif np.isclose(max_dist, dz):
            normal[2] = np.sign(z - z_c)

        return normal
