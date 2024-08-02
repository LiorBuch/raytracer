from abc import ABC, abstractmethod


class Shape(ABC):
    def __init__(self):
        self.material_index = -1

    @abstractmethod
    def get_intersection_point(self, start_point,direction):
        pass

    @abstractmethod
    def get_factor(self):
        pass
