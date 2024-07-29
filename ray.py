import numpy as np
from typing import List

from light import Light
from material import Material
from surfaces.shape import Shape


class Ray:
    def __init__(self, x, y, direction, start_position):
        self.ref_obj = None
        self.sub = None  # secondary ray if glass
        self.hit_coords = []  # on scene hit
        self.pixel_coords = (x, y)  # image pixel location
        self.direction = direction  # ray direction
        self.pos = start_position  # rays start location

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
        hit_pos = minimizer[1]#minimizer[0].get_intersection_point(self)  # TODO we also need to find the angle of the hit
        # process the hit here --->
        diffusive_color = np.array([0.0,0.0,0.0])
        specular_color = np.array([0.0,0.0,0.0])
        bg_color = np.array([1.0,1.0,1.0])
        for light in lights:
            # diffuse
            diffusive_color += np.array(self.diffuse_color(light, materials[minimizer[0].material_index-1]))*255
            # specular the normal is not good
            specular_color += np.array(self.specular_color(self.pos,np.array([0, 1, 0]),light,materials[minimizer[0].material_index-1],hit_pos))

        # phong
        # shade
        # snells
        # continue to the next ray if needed
        return self.pixel_coords , (bg_color*materials[minimizer[0].material_index-1].transparency+(diffusive_color+specular_color)*(1-materials[minimizer[0].material_index-1].transparency)) #self.pixel_coords

    def snells_law(self,hit_pos,enter:bool,obj):
        glass_media = 1.58
        air_media = 1.0003
        n= (obj.position - hit_pos) / np.linalg.norm(obj.position - hit_pos) # normal pointing in the object
        i = self.direction / np.linalg.norm(self.direction) # vector of the ray
        #t is the result vector
        mu= glass_media/air_media
        if enter:
            mu = 1/mu
        t = np.sqrt(1-(mu**2)*(1-(np.dot(n,i)**2)))
        t = t / np.linalg.norm(t)

        return hit_pos , t
    def diffuse_color(self,light:Light,hit_obj_mat:Material):
        color = hit_obj_mat.diffuse_color * light.color
        return color

    def specular_color(self, view_pos, surface_normal , light:Light,mat:Material,hit_pos):
        def normalize(v):
            norm = np.linalg.norm(v)
            return v / norm if norm != 0 else v

        # Normalize vectors
        light_vec = normalize(light.position - hit_pos)
        view_vec = normalize(view_pos - hit_pos)
        normal_vec = normalize(surface_normal)

        # Compute reflection vector R
        reflection_vec = 2 * np.dot(light_vec, normal_vec) * normal_vec - light_vec
        reflection_vec = normalize(reflection_vec)

        # Compute dot product R.V and clamp to 0 if negative
        dot_rv = max(np.dot(reflection_vec, view_vec), 0)

        # Calculate the specular component
        specular_component = mat.specular_color * (dot_rv ** mat.shininess)

        # Specular color
        specular_color = light.color * specular_component

        return specular_color

    def shadow_ray(self):
        pass

    def compute_triangle(self):
        pass

    def compute_sphere(self):
        pass

    def compute_plane(self):
        pass

    def glass_hit(self):
        pass
