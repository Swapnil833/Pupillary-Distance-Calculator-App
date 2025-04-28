from line import *

class Pupil:
  def __init__(self, point_a, point_b, point_c, point_d):
    self.line_r = Line(Point(point_a[0], point_a[1]),
        Point(point_b[0], point_b[1]))

    self.line_s = Line(Point(point_c[0], point_c[1]),
        Point(point_d[0], point_d[1]))

    self.central_point = self.get_pupil_coordinate()

  def get_pupil_coordinate(self):
    central_point = Line.intersection(self.line_r, self.line_s)
    return central_point
  
  def distance(left_pupil, right_pupil):
    # Euclidean distance
    distance = ((right_pupil.x - left_pupil.x)**2 + (right_pupil.y - left_pupil.y)**2)
    return distance

class Eyelid:
  def __init__(self, point_a, point_b):
    self.point_1 = Point(point_a[0], point_a[1])
    self.point_2 = Point(point_b[0], point_b[1]) 
    
    self.midpoint = self.get_MRD_coordinates()
    
  def get_MRD_coordinates(self):
    midpoint = Line.get_midpoint(self.point_1, self.point_2)
    return midpoint