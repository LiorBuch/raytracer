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
    def __init__(self, x, y, direction, start_position, max_rec, max_shadow, camera_pos, background_color):
        self.hit_coords = []  # on scene hit
        self.pixel_coords = (x, y)  # image pixel location
        self.direction = direction  # ray direction
        self.pos = start_position  # rays start location
        self.max_shadow_rays = int(max_shadow)
        self.max_recursions = max_rec  # max_rec
        self.camera_pos = camera_pos
        self.background_color = np.array(background_color)

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
                    # Assuming that each hit object has another distance
                    options[distance] = (obj, pos)
        if 0 == len(options.keys()):  # If didn't hit any object, return the background color defined.
            return self.pixel_coords, self.background_color

        minimizer = options[min(options.keys())]  # The closest object
        hit_obj = minimizer[0]
        hit_pos = np.array(minimizer[1])
        obj_mat = materials[hit_obj.material_index - 1]
        self.direction = self.normalize(hit_pos - self.pos)  # From camera
        normal = self.get_normal(hit_obj, hit_pos)
        # process the hit here --->
        total_color = np.array([0.0, 0.0, 0.0])
        for light in lights:  # Lets shoot rays from each light to the object and caclulate the light in that coordinate!
            light_dir = self.normalize(light.position - hit_pos)
            shadow_factor = self.calculate_soft_shadows(hit_pos, light.position, light.radius, self.max_shadow_rays,
                                                        objects, light.shadow_intensity, hit_obj, materials)

            diffusive_color = np.array(self.diffuse_color(normal, light_dir, light.color, obj_mat.diffuse_color))
            specular_color = np.array(
                self.specular_color(normal, light_dir, self.normalize(self.pos - hit_pos), obj_mat.shininess,
                                    obj_mat.specular_color,
                                    light.specular_intensity, light.color))
            total_color += (specular_color + diffusive_color) * shadow_factor * (1 - obj_mat.transparency)

        if obj_mat.transparency != 0 and self.max_recursions > 0:
            # Using snell's law for transparency! We decided the n as we liked.
            snell_dir, exit_pos = self.snells_law(self.direction, hit_pos, normal, hit_obj)
            transparent_ray = Ray(self.pixel_coords[0], self.pixel_coords[1], self.normalize(snell_dir), exit_pos,
                                  self.max_recursions - 1,
                                  self.max_shadow_rays, self.camera_pos, self.background_color)
            _, transparent_color = transparent_ray.shoot(objects, lights, materials, hit_obj)
            transparent_color *= obj_mat.transparency
            total_color += transparent_color

        if self.max_recursions > 0 and np.sum(obj_mat.reflection_color) != 0:
            next_direction = self.reflect(-self.direction, normal)
            next_ray = Ray(self.pixel_coords[0], self.pixel_coords[1], next_direction, hit_pos, self.max_recursions - 1,
                           self.max_shadow_rays, self.camera_pos, self.background_color)
            _, reflected_color = next_ray.shoot(objects, lights, materials, hit_obj)
            total_color += obj_mat.reflection_color * reflected_color

        total_color = np.clip(total_color, 0, 1)
        return self.pixel_coords, total_color

    def get_normal(self, obj, hit_vec):
        # Each object has different way to calc it's normal
        if (isinstance(obj, Sphere)):
            return self.normalize(hit_vec - obj.position)
        if (isinstance(obj, Cube)):
            return obj.get_normal(hit_vec)
        if (isinstance(obj, InfinitePlane)):
            return self.normalize(obj.normal)

    def snells_law(self, hit_vec, hit_pos, normal, obj):
        glass_media = 1.58
        air_media = 1.0003

        mu_in = air_media / glass_media
        angle_in = np.sqrt(1 - (mu_in ** 2) * (1 - np.dot(-hit_vec, normal) ** 2))
        refract_in = mu_in * hit_vec + (mu_in * np.dot(-hit_vec, normal) - angle_in) * normal

        hit, exit_pos = obj.get_intersection_point(hit_pos - normal * 0.001, refract_in)
        if not hit:
            return refract_in, hit_pos

        # exit_ray
        mu_out = 1 / mu_in
        refract_in = self.normalize(refract_in)
        angle_out = np.sqrt(1 - (mu_out ** 2) * (1 - np.dot(-refract_in, -normal) ** 2))
        refract_out = mu_out * refract_in + (mu_out * np.dot(-refract_in, -normal) - angle_out) * normal

        return refract_out, exit_pos

    def diffuse_color(self, normal, light_dir, light_color, mat_diffuse_color):
        light_dir = self.normalize(light_dir)
        intensity = max(np.dot(light_dir, normal), 0)  # So the intensity won't be negative
        color = intensity * light_color * mat_diffuse_color
        return color

    def specular_color(self, normal, light_dir, view, phong, mat_spec, light_spec, light_color):
        reflected = 2 * np.dot(normal, light_dir) * normal - light_dir
        intensity = max(0, np.dot(self.normalize(reflected), self.normalize(view))) ** phong
        specular_color = mat_spec * intensity * light_spec * light_color
        return specular_color

    def reflect(self, vector, normal):
        vector = self.normalize(vector)
        reflected = 2 * np.dot(normal, vector) * normal - vector
        return self.normalize(reflected)

    def calculate_soft_shadows(self, surface_point, light_position, light_radius, num_shadow_rays, objects,
                               shadow_intensity, test_obj, materials):
        dir_to_light_center = light_position - surface_point
        if dir_to_light_center[0] != 0 or dir_to_light_center[1] != 0:
            rand_v = np.array([0, 0, 1])
        else:
            rand_v = np.array([0, 1, 0])

        x_plane = np.cross(dir_to_light_center, rand_v)
        x_plane = self.normalize(x_plane)
        y_plane = np.cross(dir_to_light_center, x_plane)
        y_plane = self.normalize(y_plane)
        partly_shade = 0
        grid_size = light_radius * 2 / num_shadow_rays
        for i in range(num_shadow_rays):
            for j in range(num_shadow_rays):
                x = (i + np.random.uniform(0, 1)) * grid_size - light_radius
                y = (j + np.random.uniform(0, 1)) * grid_size - light_radius
                jittered_point = light_position + x * x_plane + y * y_plane
                direction = self.normalize((jittered_point - surface_point))
                dist_to_light = np.linalg.norm(jittered_point - surface_point)
                factor = self.light_hit(objects, surface_point, direction, test_obj, materials, dist_to_light)
                partly_shade += factor
        hit_rate = (partly_shade / (num_shadow_rays ** 2))
        return (1 - shadow_intensity) + hit_rate * shadow_intensity

    def light_hit(self, objects, start_point, direction, test_obj, materials, dist_to_light):
        # Searching if there is an object that blocks the light.
        for obj in objects:
            if test_obj != obj:
                hit, pos = obj.get_intersection_point(np.array(start_point), np.array(direction))
                if hit:
                    # If the object we hit is behind the light, it doesn't count as hitting an object - it doesn't block the light!
                    if dist_to_light <= np.linalg.norm(pos - start_point):
                        pass
                    return materials[obj.material_index - 1].transparency
        # The ray reached the light successfully!
        return 1

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v
