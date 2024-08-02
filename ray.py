import numpy as np
from typing import List

import numpy.random.mtrand

from light import Light
from material import Material
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.shape import Shape
from surfaces.sphere import Sphere


class Ray:
    def __init__(self, x, y, direction, start_position, max_rec, max_shadow):
        self.ref_obj = None
        self.sub = None  # secondary ray if glass
        self.hit_coords = []  # on scene hit
        self.pixel_coords = (x, y)  # image pixel location
        self.direction = direction  # ray direction
        self.pos = start_position  # rays start location
        self.max_shadow_rays = max_shadow
        self.max_recursions = max_rec

    def shoot(self, objects: List[Shape], lights: List[Light], materials: List[Material]):
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
            return self.pixel_coords, (0, 0, 0)

        minimizer = options[min(options.keys())]
        hit_obj = minimizer[0]
        hit_pos = minimizer[1]
        normal = self.get_normal(hit_obj, self.direction)

        # process the hit here --->
        total_color = np.array([0.0, 0.0, 0.0])
        diffusive_color = np.array([0.0, 0.0, 0.0])
        ambient_color = np.array([0.0, 0.0, 0.0])
        specular_color = np.array([0.0, 0.0, 0.0])
        bg_color = np.array([1.0, 1.0, 1.0])
        obj_mat = materials[hit_obj.material_index - 1]
        max_lights = len(lights)
        for light in lights:
            light_dir = self.normalize(light.position - hit_pos)
            ambient_intensity = 0.1
            # ambient
            ambient_color += np.array(self.ambient(ambient_intensity, obj_mat.diffuse_color))
            # diffuse
            diffusive_color += np.array(self.diffuse_color(normal, light_dir, light.color, obj_mat.diffuse_color))
            # specular
            specular_color += np.array(
                self.specular_color(normal, light_dir, -self.direction, obj_mat.shininess, light.color,
                                    obj_mat.specular_color,
                                    light.specular_intensity))
            shadow_factor = self.shadow_ray(hit_pos,light.radius,light.position)
            shadow_factor/=max_lights
            total_color += ambient_color + shadow_factor * (diffusive_color + specular_color)

        # phong
        # shade
        # snells
        # continue to the next ray if needed
        total_color = np.clip(total_color, 0, 1) * 255
        return self.pixel_coords, total_color

    def get_normal(self, obj, hit_vec):
        if (isinstance(obj, Sphere)):
            return self.normalize(obj.position - hit_vec)
        if (isinstance(obj, Cube)):
            pass  # TODO implement cube hit normal
        if (isinstance(obj, InfinitePlane)):
            return self.normalize(obj.normal)

    def snells_law(self, hit_pos, enter: bool, obj):
        glass_media = 1.58
        air_media = 1.0003
        n = (obj.position - hit_pos) / np.linalg.norm(obj.position - hit_pos)  # normal pointing in the object
        i = self.direction / np.linalg.norm(self.direction)  # vector of the ray
        # t is the result vector
        mu = glass_media / air_media
        if enter:
            mu = 1 / mu
        t = np.sqrt(1 - (mu ** 2) * (1 - (np.dot(n, i) ** 2)))
        t = t / np.linalg.norm(t)

        return hit_pos, t

    def diffuse_color(self, normal, light_dir, light_color, mat_diffuse_color):
        intensity = max(np.dot(normal, light_dir), 0)
        color = intensity * light_color * mat_diffuse_color
        return color

    def specular_color(self, normal, light_dir, view, phong, light_color, mat_spec, light_spec_intensity):
        reflect_dir = self.normalize(2 * normal * np.dot(normal, light_dir) - light_dir)
        intensity = max(np.dot(self.normalize(view), reflect_dir), 0) ** phong
        specular_color = intensity * light_color * mat_spec * light_spec_intensity
        return specular_color

    def ambient(self, intensity, diffuse_color):
        return intensity * diffuse_color

    def shadow_ray(self, hit_pos, light_radius, light_center):
        x_plane = np.random.randn(3)
        dir_to_light_center = self.normalize(light_center - hit_pos)
        x_plane -= x_plane.dot(dir_to_light_center) * dir_to_light_center
        x_plane = self.normalize(x_plane)
        y_plane = self.normalize(np.cross(dir_to_light_center, x_plane))
        # x_plane,y_plane generate the plane with the maximum hit rate
        hit = 0
        for _ in range(int(self.max_shadow_rays)):
            # Random angle between 0 and 2*pi
            theta = np.random.uniform(0, 2 * np.pi)
            # Random radius with uniform distribution within the circle
            r = light_radius * np.sqrt(np.random.uniform(0, 1))

            # Point in the plane
            point = light_center + r * (np.cos(theta) * x_plane + np.sin(theta) * y_plane)
            happen, _ =self.light_hit(hit_pos, point - hit_pos, light_center, light_radius)
            if happen:
                hit += 1
        return hit / self.max_shadow_rays

    @staticmethod
    def light_hit(start_point, direction, light_pos, light_radius) -> (bool, np.array):
        # According to lecture 7, page 39
        a = 1
        b = 2 * np.dot(direction, (start_point - light_pos))
        c = np.linalg.norm(start_point - light_pos) ** 2 - light_radius ** 2
        det = b ** 2 - 4 * a * c
        if det < 0:
            return False, np.array([0, 0, 0])
        t1 = (-b + det ** 0.5) / (2 * a)
        t2 = (-b - det ** 0.5) / (2 * a)
        t = min(t1, t2)
        if t < 0:
            return False, np.array([0, 0, 0])
        return True, start_point + direction * t

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v
