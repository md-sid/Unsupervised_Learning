import random
import math
from planar import Polygon
from opt_center_points import OptimizedCenters


def polygon_shape(num_poly, x_range, y_range):
    x_dist = x_range[1] - x_range[0]
    y_dist = y_range[1] - y_range[0]
    area = x_dist * y_dist
    max_area = area / num_poly
    max_radius = int(math.sqrt(max_area / math.pi))
    # print(max_radius)

    center_x_range = [x_range[0] + int(max_radius * 0.6), x_range[1] - int(max_radius * 0.6)]
    center_y_range = [y_range[0] + int(max_radius * 0.6), y_range[1] - int(max_radius * 0.6)]
    
    a = OptimizedCenters(center_x_range, center_y_range, max_radius, num_poly)
    a.best_option()
    random_centers = a.points
    radius = a.radius
    # print(radius)

    polygon = []
    for index in range(num_poly):
        poly_sides = random.randint(3, 10)
        poly_radius = radius
        tmp = Polygon.regular(poly_sides, poly_radius, random_centers[index], angle=poly_sides * 10)
        polygon.append(tmp)
        index += 1
    return polygon, random_centers, radius
