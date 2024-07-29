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


class SceneBuilder:
    def __init__(self, camera: Camera, scene_settings: SceneSettings, objects: list):

        self.camera = camera
        self.max_workers = 6
        self.batch = 9
        self.scene_settings = scene_settings
        self.objects = [obj for obj in objects if isinstance(obj, Shape)]
        self.lights = [light for light in objects if isinstance(light, Light)]
        self.materials = [mat for mat in objects if isinstance(mat, Material)]
        self.voxel_grid = None
        self.pop_grid = None
        self.width = 10  # int(self.camera.screen_width)
        self.height = 10  # TODO figure out aspect
        self.create_subdivision_grid()
        manager = mp.Manager()
        self.iterations = manager.Value('i', 0)
        self.lock = manager.Lock()
        print(f"time for building scene: {9.5 * self.width * self.height / 100}")

    def create_scene(self) -> np.array:
        img = np.zeros((self.height, self.width, 3))

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = [executor.submit(self.ray_task, int(i / self.width), i % self.width, self.lock, self.iterations)
                     for i in range(self.height * self.width)]
            for future in concurrent.futures.as_completed(tasks):
                x, y, color = future.result()
                img[x, y, :] = color

        return img

    def create_scene_batch(self) -> np.array:
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
            tasks = [executor.submit(self.ray_task_batched, param[k][0],param[k][1] ,param[k][2]) for k in range(len(param))]
            for future in concurrent.futures.as_completed(tasks):
                data, (batch_s, batch_e) = future.result()
                color = np.array(data[:])
                img[batch_s:batch_e, :] = color
        img.resize((self.width, self.height, 3))
        return img

    def cuda_create_scene(self):
        materials = np.array(self.materials)
        img = np.zeros((self.height, self.width, 3), dtype=np.float32)

        d_image = cuda.to_device(img)
        d_materials = cuda.to_device(materials)

        threads_per_block = (16, 16)
        blocks_per_grid = (self.width // threads_per_block[0] + 1, self.height // threads_per_block[1] + 1)

        self.cuda_ray_task[blocks_per_grid, threads_per_block](d_image, d_materials)

        d_image.copy_to_host(img)
        return img

    @staticmethod
    @cuda.jit
    def cuda_ray_task(img, materials):
        x, y = cuda.grid(2)
        if x >= img.shape[1] or y >= img.shape[0]:
            return

        ray_origin = np.array([0, 0, 0], dtype=np.float32)  # Example camera position
        ray_direction = np.array([x - img.shape[1] / 2, y - img.shape[0] / 2, 1], dtype=np.float32)
        ray_direction /= np.linalg.norm(ray_direction)
        color = np.zeros(3, dtype=np.float32)

        # Iterate through objects in the scene
        for obj_idx in range(len(materials)):  # Assuming each material corresponds to an object
            # Example: replace with actual intersection calculation
            t = obj_idx  # This is just a placeholder. Replace with actual intersection check.
            if t:
                color = materials[obj_idx]
                break  # Break on first hit for simplicity

        img[y, x] = color

    @staticmethod
    def intersect_sphere(ray_origin, ray_direction, sphere):
        # Calculate intersection
        oc = ray_origin - sphere.center
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - sphere.radius * sphere.radius
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None
        else:
            t = (-b - np.sqrt(discriminant)) / (2.0 * a)
            if t < 0:
                return None
            return t

    def ray_task(self, pixel_i, pixel_j, lock, iterations):

        camera_dir = self.camera.look_at - self.camera.position
        camera_dir /= np.linalg.norm(camera_dir)
        camera_dir *= self.camera.screen_distance

        height_dir = (self.camera.up_vector) * ((self.height / 2) - pixel_i)
        width_dir = -np.linalg.cross(camera_dir, self.camera.up_vector)
        width_dir = (width_dir / np.linalg.norm(width_dir)) * ((self.width / 2) - pixel_j)
        screen_dir = height_dir + width_dir
        pixel_dir = camera_dir + screen_dir
        ray: Ray = Ray(pixel_i, pixel_j, pixel_dir, self.camera.position)
        (x, y), rgb = ray.shoot(self.objects, self.lights, self.materials)
        with lock:
            iterations.value += 1
            # print(f"Calculated {rgb}")
            # print(f"Direction -> {pixel_dir}")
            if iterations.value % 100 == 0:
                print(f"pixels left: {self.width * self.height - iterations.value}")
            # print("")

        return pixel_i, pixel_j, rgb

    def ray_task_batched(self, params, s, e):
        camera_dir = self.camera.look_at - self.camera.position
        camera_dir /= np.linalg.norm(camera_dir)
        camera_dir *= self.camera.screen_distance
        data = []
        for pixel_i, pixel_j in params:
            height_dir = (self.camera.up_vector) * ((self.height / 2) - pixel_i)
            width_dir = -np.linalg.cross(camera_dir, self.camera.up_vector)
            width_dir = (width_dir / np.linalg.norm(width_dir)) * ((self.width / 2) - pixel_j)
            screen_dir = height_dir + width_dir
            pixel_dir = camera_dir + screen_dir
            ray: Ray = Ray(pixel_i, pixel_j, pixel_dir, self.camera.position)
            data.append(ray.shoot(self.objects, self.lights, self.materials)[1])
        print("batch ready")
        return data, (s, e)

    # https://developer.nvidia.com/gpugems/gpugems2/part-i-geometric-complexity/chapter-7-adaptive-tessellation-subdivision-surfaces#:~:text=Adaptive%20Subdivision&text=Instead%20of%20blindly%20subdividing%20a,the%20more%20it%20gets%20subdivided.
    def create_subdivision_grid(self):
        # first we find the min x,y,z and max x,y,z
        min_x, min_y, min_z = (-1, -1, -1)
        max_x, max_y, max_z = (-1, -1, -1)
        for obj in self.objects:
            if isinstance(obj, InfinitePlane):
                continue

            test_min = np.array(obj.position) - obj.get_factor()
            test_max = np.array(obj.position) + obj.get_factor()
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
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
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
    """
    new idea, instead of 3D matrix, a flat list of the voxels, each element in the list will represent x,y,z and
    delta(x,y,z)
    they dont need to be aligned, yet the problem is finding the next voxel.
    """

    def divide_voxel(self, voxel, dx, dy, dz):
        sub_grid = voxel
        for obj in self.objects:
            if obj.position[0] in (voxel[0], voxel[0] + dx) and obj.position[1] in (voxel[1], voxel[1] + dy) and \
                    obj.position[2] in (voxel[2], voxel[2] + dz):
                return

    def populate_grid(self, voxel, dx, dy, dz):
        pop_list = []
        for obj in self.objects:
            if isinstance(obj, Cube):
                cube_max_x = obj.position[0] + obj.scale
                cube_max_y = obj.position[1] + obj.scale
                cube_max_z = obj.position[2] + obj.scale
                cube_min_x = obj.position[0] - obj.scale
                cube_min_y = obj.position[1] - obj.scale
                cube_min_z = obj.position[2] - obj.scale
                if self.is_overlapping((cube_min_x, cube_min_y, cube_min_z), (cube_max_x, cube_max_y, cube_max_z),
                                       voxel, (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)):
                    pop_list.append(obj)
            elif isinstance(obj, Sphere):
                sphere_max_x = obj.position[0] + obj.radius
                sphere_max_y = obj.position[1] + obj.radius
                sphere_max_z = obj.position[2] + obj.radius
                sphere_min_x = obj.position[0] - obj.radius
                sphere_min_y = obj.position[1] - obj.radius
                sphere_min_z = obj.position[2] - obj.radius
                if self.is_overlapping((sphere_min_x, sphere_min_y, sphere_min_z),
                                       (sphere_max_x, sphere_max_y, sphere_max_z),
                                       voxel, (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)):
                    pop_list.append(obj)
            elif isinstance(obj, InfinitePlane):
                # If the normal of the plane * vector from the camera to that plane = 0,
                # the plane will not be seen in the final image -> So consider adding this logic
                point_on_plane = obj.get_point_on_plane()
                if np.dot(point_on_plane - self.camera.position, obj.normal) != 0:
                    pop_list.append(obj)
        return pop_list

    def is_overlapping(self, a_min, a_max, b_min, b_max):
        if a_max[0] < b_min[0] or a_min[0] > b_max[0]:
            return False
        if a_max[1] < b_min[1] or a_min[1] > b_max[1]:
            return False
        if a_max[2] < b_min[2] or a_min[2] > b_max[2]:
            return False
        return True
