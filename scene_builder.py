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
        self.voxel_grid = None
        self.pop_grid = None
        self.width = self.camera.screen_width
        self.height = 200  # TODO figure out aspect

    def create_scene(self) -> np.array:
        img = np.zeros(())
        width = self.width
        height = self.height

        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
            tasks = [executor.submit(self.ray_task, int(i / height), i % width) for i in range(height * width)]
            for future in concurrent.futures.as_completed(tasks):
                x, y, color = future.result()
                img[x, y, :] = color

    def ray_task(self, pixel_i, pixel_j):
        camera_dir = np.linalg.norm(self.camera.look_at - self.camera.position)
        screen_dir = np.linalg.norm((self.height / 2, self.width / 2) - (pixel_i, pixel_j))
        pixel_dir = camera_dir + screen_dir
        ray: Ray = Ray(pixel_i, pixel_j, pixel_dir, self.camera.position)
        rgb = ray.shoot(self.objects)
        return pixel_i, pixel_j, rgb

    # https://developer.nvidia.com/gpugems/gpugems2/part-i-geometric-complexity/chapter-7-adaptive-tessellation-subdivision-surfaces#:~:text=Adaptive%20Subdivision&text=Instead%20of%20blindly%20subdividing%20a,the%20more%20it%20gets%20subdivided.
    def create_subdivision_grid(self):
        # first we find the min x,y,z and max x,y,z
        min_x, min_y, min_z = (-1, -1, -1)
        max_x, max_y, max_z = (-1, -1, -1)
        for obj in self.objects:
            test_min = obj.position - obj.radius
            test_max = obj.position + obj.radius
            if (test_min[0] < min_x):
                min_x = test_min[0]
            if (test_min[1] < min_y):
                min_y = test_min[1]
            if (test_min[2] < min_z):
                min_z = test_min[2]

            if (test_max[0] < max_x):
                max_x = test_max[0]
            if (test_max[1] < max_y):
                max_y = test_max[1]
            if (test_max[2] < max_z):
                max_z = test_max[2]
        # generate bounding box
        division_factor = 20
        grid = [[[[min_x + (max_x - min_x) * i / division_factor, min_y + (max_x - min_y) * j / division_factor,
                   min_z + (max_z - min_z) * k / division_factor]
                  for i in range(0, division_factor + 1)] for j in range(0, division_factor + 1)] for k in
                range(0, division_factor + 1)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
            flat_grid = [voxel for subgrid1 in grid for subgrid2 in subgrid1 for voxel in subgrid2]
            futures = [
                executor.submit(self.populate_grid, voxel, max_x - min_x, max_y - min_y, max_z - min_z) for voxel in
                flat_grid]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        index = 0
        pop_list = [[[[] for i in range(0, division_factor + 1)] for j in range(0, division_factor + 1)] for k in
                    range(0, division_factor + 1)]
        for k in range(len(grid)):
            for j in range(len(grid[k])):
                for i in range(len(grid[k][j])):
                    pop_list[k][j][i] = results[index]
                    index += 1
        self.voxel_grid = grid
        self.pop_grid = pop_list

    # not in use for now
    def divide_voxel(self, voxel, dx, dy, dz):
        sub_grid = voxel
        for obj in self.objects:
            if obj.position[0] in (voxel[0], voxel[0] + dx) and obj.position[1] in (voxel[1], voxel[1] + dy) and \
                    obj.position[2] in (voxel[2], voxel[2] + dz):
                return

    def populate_grid(self, voxel, dx, dy, dz):
        pop_list = []
        for obj in self.objects:
            if obj.position[2] in (voxel[2], voxel[2] + dz) and obj.position[0] in (voxel[0], voxel[0] + dx) and \
                    obj.position[1] in (voxel[1], voxel[1] + dy):
                pop_list.append(obj)
        return pop_list
