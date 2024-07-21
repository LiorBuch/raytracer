import numpy as np
from camera import Camera
from ray import Ray
from scene_settings import SceneSettings
import concurrent.futures


class SceneBuilder:
    def __init__(self, camera: Camera, scene_settings: SceneSettings, objects: list):
        self.camera = camera
        self.scene_settings = scene_settings
        self.objects = objects
        self.width = self.camera.screen_width
        self.height = 200  # TODO figure out aspect

    def create_scene(self) -> np.array:
        img = np.zeros(())
        width = self.width
        height = self.height

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            tasks = [executor.submit(self.ray_task, int(i / height), i % width) for i in range(height * width)]
            for future in concurrent.futures.as_completed(tasks):
                x, y, color = future.result()
                img[x, y, :] = color

    def ray_task(self, pixel_i, pixel_j):
        camera_dir = np.linalg.norm(self.camera.look_at - self.camera.position)
        screen_dir = np.linalg.norm((self.height / 2, self.width / 2) - (pixel_i, pixel_j))
        pixel_dir = camera_dir + screen_dir
        ray: Ray = Ray(pixel_i, pixel_j, pixel_dir, (pixel_i, pixel_j))
        rgb = ray.shoot(self.objects)
        return pixel_i,pixel_j,rgb
