import random

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
    def __init__(self, x, y, direction, start_position, max_rec, max_shadow, camera_pos):
        self.ref_obj = None
        self.sub = None  # secondary ray if glass
        self.hit_coords = []  # on scene hit
        self.pixel_coords = (x, y)  # image pixel location
        self.direction = direction  # ray direction
        self.pos = start_position  # rays start location
        self.max_shadow_rays = int(max_shadow)
        self.max_recursions = max_rec
        self.camera_pos = camera_pos

    def shoot(self, objects: List[Shape], lights: List[Light], materials: List[Material], ignore=None):
        """
        Shoots the ray and returns the hit object and the hit position
        :param objects: All relevant objects that the ray may hit
        :param lights: list of the scene lights
        :param materials: list of the scene materials
        :param ignore: object to ignore when searching
        :return: if the ray hit an object, tuple of hit object and the hit position. Otherwise, returns None
        """
        options = {}
        for obj in objects:
            if obj != ignore:
                hit, pos = obj.get_intersection_point(self.pos, self.direction)
                if hit:
                    distance = np.linalg.norm(self.pos - pos)
                    # Assuming that each hit object has another distance - maybe should change in the future
                    options[distance] = (obj, pos)
        if 0 == len(options.keys()):
            return self.pixel_coords, np.array((1.0, 1.0, 1.0))
        minimizer = options[min(options.keys())]
        hit_obj = minimizer[0]
        hit_pos = np.array(minimizer[1])
        obj_mat = materials[hit_obj.material_index - 1]
        self.direction = self.normalize(self.pos + hit_pos)
        normal = self.get_normal(hit_obj, hit_pos)
        # process the hit here --->
        total_color = np.array([0.0, 0.0, 0.0])
        bg_color = np.array([1.0, 1.0, 1.0])
        max_lights = len(lights)
        for light in lights:
            light_dir = self.normalize(light.position - hit_pos)

            shadow_factor = self.calculate_soft_shadows(hit_pos, light.position, light.radius, self.max_shadow_rays,
                                                        objects, light.shadow_intensity, hit_obj)

            diffusive_color = np.array(self.diffuse_color(normal, light_dir, light.color, obj_mat.diffuse_color))
            specular_color = np.array(
                self.specular_color(normal, light_dir, self.normalize(self.camera_pos - hit_pos), obj_mat.shininess,
                                    obj_mat.specular_color,
                                    light.specular_intensity))
            ambient_color = np.array(self.ambient(1/max_lights, obj_mat.diffuse_color, light.color))
            reflected_color = np.array([0.0, 0.0, 0.0])
            if self.max_recursions > 0 and np.sum(obj_mat.reflection_color) != 0 :
                next_direction = self.normalize(self.reflect(self.direction, normal))
                next_direction = self.normalize(next_direction)
                next_ray = Ray(self.pixel_coords[0], self.pixel_coords[1], next_direction, hit_pos, 0,
                               self.max_shadow_rays, self.camera_pos)
                _,reflected_color = next_ray.shoot(objects, lights, materials, hit_obj)
                reflected_color *= obj_mat.reflection_color
            total_color += (ambient_color + specular_color + diffusive_color) * shadow_factor + reflected_color

        # continue to the next ray if needed
        return self.pixel_coords, total_color

    def get_normal(self, obj, hit_vec):
        if (isinstance(obj, Sphere)):
            return self.normalize(hit_vec - obj.position)
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

    def ambient(self, intensity, diffuse_color, light_color):
        return intensity * diffuse_color * light_color

    def diffuse_color(self, normal, light_dir, light_color, mat_diffuse_color):
        light_dir = self.normalize(light_dir)
        intensity = max(np.dot(light_dir, normal), 0)
        color = intensity * light_color * mat_diffuse_color
        return color

    def specular_color(self, normal, light_dir, view, phong, mat_spec, light_spec):
        reflected = 2 * np.dot(normal, light_dir) * normal - light_dir
        intensity = max(0, np.dot(self.normalize(reflected), self.normalize(view))) ** (phong)
        specular_color = mat_spec * intensity * light_spec
        return specular_color

    def reflect(self, vector, normal):
        vector = self.normalize(vector)
        reflected = 2 * np.dot(normal, vector) * normal - vector
        return reflected

    def calculate_soft_shadows_old(self, surface_point, light_position, light_radius, num_shadow_rays, objects,
                                   shadow_intensity, test_obj):

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
            if self.light_hit(objects, np.array(point), self.normalize(light_dir), test_obj):
                hit_rays_count += 1

        light_intensity = (1 - shadow_intensity) + shadow_intensity * (hit_rays_count / num_shadow_rays)

        return light_intensity

    def calculate_soft_shadows(self, surface_point, light_position, light_radius, num_shadow_rays, objects,
                               shadow_intensity, test_obj):
        dir_to_light_center = light_position - surface_point
        if dir_to_light_center[0] != 0 or dir_to_light_center[1] != 0:
            rand_v = np.array([0, 0, 1])
        else:
            rand_v = np.array([0, 1, 0])

        x_plane = np.cross(dir_to_light_center, rand_v)
        x_plane = self.normalize(x_plane)
        y_plane = np.cross(dir_to_light_center, x_plane)
        y_plane = self.normalize(y_plane)
        shadow_rays = 0
        grid_size = light_radius * 2 / num_shadow_rays
        for i in range(num_shadow_rays):
            for j in range(num_shadow_rays):
                x = (i + random.uniform(0, 1)) * grid_size - light_radius
                y = (j + random.uniform(0, 1)) * grid_size - light_radius
                jittered_point = light_position + x * x_plane + y * y_plane
                direction = self.normalize((jittered_point - surface_point))
                if self.light_hit(objects, surface_point, direction, test_obj):
                    shadow_rays += 1
        hit_rate = (shadow_rays / (num_shadow_rays ** 2))
        return (1 - shadow_intensity) * hit_rate

    def light_hit(self, objects, start_point, direction, test_obj):
        for obj in objects:
            if test_obj != obj:
                hit, pos = obj.get_intersection_point(np.array(start_point), np.array(direction))
                if hit:
                    return False
        return True

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v
