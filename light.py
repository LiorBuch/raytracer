import numpy as np
class Light:
    def __init__(self, position, color, specular_intensity, shadow_intensity, radius):
        self.position = np.array(position)
        self.color = np.array(color)
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.radius = radius

