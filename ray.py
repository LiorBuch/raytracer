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
        self.max_shadow_rays = int(max_shadow)
        self.max_recursions = max_rec

    def shoot(self, objects: List[Shape], lights: List[Light], materials: List[Material]):
        """
        Shoots the ray and returns the hit object and the hit position
        :param objects: All relevant objects that the ray may hit
        :return: if the ray hit an object, tuple of hit object and the hit position. Otherwise, returns None
        """
        options = {}
        for obj in objects:
            hit, pos = obj.get_intersection_point(self.pos, self.direction)
            if hit:
                distance = np.linalg.norm(self.pos - pos)
                # Assuming that each hit object has another distance - maybe should change in the future
                options[distance] = (obj, pos)
        if 0 == len(options.keys()):
            return self.pixel_coords, np.array([0.0, 0.0, 0.0])

        minimizer = options[min(options.keys())]
        hit_obj = minimizer[0]
        hit_pos = minimizer[1]
        normal = self.get_normal(hit_obj, self.direction)

        """if hit object is light, return 1"""

        # process the hit here --->
        total_color = np.array([0.0, 0.0, 0.0])
        bg_color = np.array([1.0, 1.0, 1.0])
        obj_mat = materials[hit_obj.material_index - 1]
        max_lights = len(lights)
        for light in lights:
            light_dir = self.normalize(light.position - hit_pos)
            ambient_intensity = 0
            # ambient
            ambient_color = np.array(self.ambient(ambient_intensity, obj_mat.diffuse_color))
            # diffuse
            diffusive_color = np.array(self.diffuse_color(normal, light_dir, light.color, obj_mat.diffuse_color))
            # specular
            specular_color = np.array(
                self.specular_color(normal, light_dir, -self.direction, obj_mat.shininess, light.color,
                                    obj_mat.specular_color,
                                    light.specular_intensity))
            reflective_color = np.array([0.0, 0.0, 0.0])
            shadow_factor = self.calculate_soft_shadows(hit_pos, light.position, light.radius, self.max_shadow_rays,
                                                        objects, min(options.keys()), light.shadow_intensity)
            shadow_factor /= max_lights
            total_color += ambient_color + bg_color * obj_mat.transparency + shadow_factor * (
                    diffusive_color + specular_color) * (
                                       1 - obj_mat.transparency)  # TODO how to get the correct color for reflection
            if (self.max_recursions > 0 and np.sum(obj_mat.reflection_color) != 0):
                reflect_ray = Ray(self.pixel_coords[0], self.pixel_coords[1], self.reflect(self.direction, normal),
                                  hit_pos, 0, self.max_shadow_rays)
                reflective_color += np.array(reflect_ray.shoot(objects, lights, materials)[1])
                total_color += reflective_color * obj_mat.reflection_color
            return self.pixel_coords, total_color

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

    def reflect(self, incident, normal):
        return incident - 2 * np.dot(normal, incident) * normal

    def specular_color(self, normal, light_dir, view, phong, light_color, mat_spec, light_spec_intensity):
        normal = self.normalize(normal)
        light_dir = self.normalize(light_dir)
        view = self.normalize(view)

        reflect_dir = self.normalize(2 * normal * np.dot(normal, light_dir) - light_dir)
        intensity = max(np.dot(view, reflect_dir), 0) ** phong

        specular_color = intensity * light_color * mat_spec * light_spec_intensity

        return specular_color

    def ambient(self, intensity, diffuse_color):
        return intensity * diffuse_color

    def calculate_soft_shadows(self, surface_point, light_position, light_radius, num_shadow_rays, objects, min_dist,
                               shadow_intensity):

        x_plane = np.random.randn(3)
        dir_to_light_center = self.normalize(light_position - surface_point)
        x_plane -= x_plane.dot(dir_to_light_center) * dir_to_light_center
        x_plane = self.normalize(x_plane)
        y_plane = self.normalize(np.cross(dir_to_light_center, x_plane))

        x = np.random.uniform(light_position - x_plane * light_radius, light_position + x_plane * light_radius,
                              size=(self.max_shadow_rays, 3))
        y = np.random.uniform(light_position - y_plane * light_radius, light_position + y_plane * light_radius,
                              size=(self.max_shadow_rays, 3))
        factor = np.random.uniform(0, 1, size=(self.max_shadow_rays, 1))
        min_factor = np.ones((self.max_shadow_rays, 1)) - factor
        x_factor = factor * x
        y_factor = min_factor * y
        rand_light_point = x_factor + y_factor
        light_to_surface = rand_light_point - surface_point

        light_list = rand_light_point.tolist()
        light_dir_list = light_to_surface.tolist()
        hit_rays_count = 0
        for point, light_dir in zip(light_list, light_dir_list):
            if self.light_hit(objects, min_dist, np.array(point), self.normalize(light_dir)):
                hit_rays_count += 1

        light_intensity = (1 - shadow_intensity) + shadow_intensity * (hit_rays_count / num_shadow_rays)
        return light_intensity

    def light_hit(self, objects, min_dist, start_point, direction):
        for obj in objects:
            hit, pos = obj.get_intersection_point(np.array(start_point), np.array(direction))
            if hit:
                distance = np.linalg.norm(self.pos - pos)
                if distance < min_dist:
                    return False
        return True

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v
