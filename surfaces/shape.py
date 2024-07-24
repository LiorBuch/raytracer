from abc import  ABC,abstractmethod

from ray import Ray


class Shape(ABC):
    @abstractmethod
    def get_intersection_point(self,ray:Ray):
        pass
    @abstractmethod
    def get_factor(self):
        pass
