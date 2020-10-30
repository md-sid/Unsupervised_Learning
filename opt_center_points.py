import random
import itertools
import math


def calculate_distance(point1, point2):
    x = (point1[0] - point2[0]) ** 2
    y = (point1[1] - point2[1]) ** 2
    dist = math.sqrt(x + y)
    return dist

class OptimizedCenters:

    def __init__(self, x_range, y_range, max_radius, num_centers):
        self.points = []
        self.radius = []
        self.x_range = x_range
        self.y_range = y_range
        self.max_radius = max_radius
        self.num_centers = num_centers
        self.num_iter = 10000
        self.centers = []

    def all_centerpoints(self):
        self.centers = []
        for i in range(self.num_iter):
            self.centers.append([])
            j = 0
            while j < self.num_centers:
                x = random.randrange(*self.x_range)
                y = random.randrange(*self.y_range)
                self.centers[i].append((x, y))
                j += 1
            i += 1
        return self.centers

    def all_pairs(self):
        tmp_list = list(itertools.combinations(range(self.num_centers), 2))
        return tmp_list

    def find_all_dist(self):
        pair_list = self.all_pairs()
        center_list = self.all_centerpoints()
        distance = []
        for i in range(self.num_iter):
            tmp_center = center_list[i]
            distance.append([])
            for pair in range(len(pair_list)):
                tmp = pair_list[pair]
                x_idx = tmp[0]
                y_idx = tmp[1]
                x1 = tmp_center[x_idx][0]
                x2 = tmp_center[y_idx][0]
                y1 = tmp_center[x_idx][1]
                y2 = tmp_center[y_idx][1]
                val = calculate_distance([x1, y1], [x2, y2])
                distance[i].append(val)
        return distance

    def best_option(self):
        dist = self.find_all_dist()
        min_dist = []
        for i in range(len(dist)):
            tmp = dist[i]
            min_dist.append(min(tmp))

        max_dist = max(min_dist)
        # print(max_dist)
        max_dist_idx = min_dist.index(max_dist)

        self.points = self.centers[max_dist_idx]
        self.radius = max_dist / 2
        
        if self.radius > self.max_radius:
            self.radius = self.max_radius
