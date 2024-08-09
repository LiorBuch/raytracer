import os

import numpy as np
from camera import Camera
from ray import Ray
from scene_settings import SceneSettings
import concurrent.futures
import multiprocessing as mp
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from surfaces.shape import Shape

from numba import cuda
from light import Light
from material import Material


class BatchedSceneBuilder:
    def __init__(self, camera: Camera, scene_settings: SceneSettings, objects: list, img_width, img_height):

        self.camera = camera
        self.max_workers = 20  # os.cpu_count()
        self.batch = 1000
        self.scene_settings = scene_settings
        self.objects = [obj for obj in objects if isinstance(obj, Shape)]
        self.lights = [light for light in objects if isinstance(light, Light)]
        self.materials = [mat for mat in objects if isinstance(mat, Material)]
        self.voxel_grid = None
        self.pop_grid = None
        self.width = img_width
        self.height = img_height
        self.total_pixels = self.width * self.height
        manager = mp.Manager()
        self.iterations = manager.Value('i', 0)
        self.lock = manager.Lock()

    def print_info(self):
        print("<---- Batch Ray Tracing Info ---->")

        print(f"Batch size: {self.batch}")
        print(f"Total pixels: {self.width * self.height}")
        print(f"Total Batches: {self.width * self.height / self.batch}")
        print(f"Max workers:{self.max_workers}")
        print(f"Batches per worker: {(self.width * self.height / self.batch) / self.max_workers}")
        print("<-------- Scene Settings -------->")
        print(f"Max recursions:{self.scene_settings.max_recursions}")
        print(f"Max shadow rays:{self.scene_settings.root_number_shadow_rays}")
        print("<-------------------------------->\n")

    def create_scene_batch(self) -> np.array:
        self.print_info()
        num_pixels = self.width * self.height
        img = np.zeros((num_pixels, 3))
        num_of_full_blocks = num_pixels // self.batch
        left_over_block = num_pixels % self.batch
        param = []
        for block in range(num_of_full_blocks):
            param_batch = []
            for i in range(block * self.batch, (block + 1) * self.batch, 1):
                param_batch.append([int(i / self.width), i % self.width])
            param.append([param_batch, block * self.batch, (block + 1) * self.batch])
        if left_over_block != 0:
            param_batch = []
            for i in range(num_of_full_blocks * self.batch, num_of_full_blocks * self.batch + left_over_block, 1):
                param_batch.append([int(i / self.width), i % self.width])
            param.append(
                [param_batch, num_of_full_blocks * self.batch, num_of_full_blocks * self.batch + left_over_block])

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = [executor.submit(self.ray_task_batched, param[k][0], param[k][1], param[k][2]) for k in
                     range(len(param))]
            for future in concurrent.futures.as_completed(tasks):
                data, (batch_s, batch_e) = future.result()
                color = np.array(data[:])
                img[batch_s:batch_e, :] = np.clip(color, 0, 1) * 255
        img.resize((self.width, self.height, 3))
        return img

    def ray_task_batched(self, params, s, e):
        def normalize(v):
            norm = np.linalg.norm(v)
            return v / norm if norm != 0 else v

        camera_dir = normalize(self.camera.look_at - self.camera.position)
        up_vector = normalize(np.cross(camera_dir, self.camera.up_vector))
        right_vector = normalize(np.cross(camera_dir, up_vector))
        screen_center = self.camera.position + camera_dir * self.camera.screen_distance
        screen_height = self.camera.screen_width * (self.width / self.height)

        data = []
        for pixel_i, pixel_j in params:
            ndc_x = (pixel_i + 0.5) / self.width  # image width
            ndc_y = (pixel_j + 0.5) / self.height  # image height

            # Screen coordinates [-0.5, 0.5]
            screen_x = (2 * ndc_x - 1) * self.camera.screen_width / 2
            screen_y = (1 - 2 * ndc_y) * screen_height / 2

            # Point on the screen in 3D space
            point_on_screen = screen_center + screen_x * right_vector + screen_y * up_vector

            # Direction vector for the pixel
            pixel_dir = normalize(point_on_screen - self.camera.position)
            ray: Ray = Ray(pixel_i, pixel_j, pixel_dir, self.camera.position, self.scene_settings.max_recursions,
                           self.scene_settings.root_number_shadow_rays, self.camera.position,
                           self.scene_settings.background_color)
            data.append(ray.shoot(self.objects, self.lights, self.materials)[1])

        return data, (s, e)
