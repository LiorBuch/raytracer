from abc import ABC, abstractmethod


class Shape(ABC):
    def __init__(self):
        self.material_index = -1

    @abstractmethod
    def get_intersection_point(self, ray):
        pass

    @abstractmethod
    def get_factor(self):
        pass
