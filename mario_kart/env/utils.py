import numpy as np

def line_intersection(p1, p2, p3, p4):

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # compute determinants
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-6:  # parallel lines or too close
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        # intersection point
        intersection_x = x1 + t * (x2 - x1)
        intersection_y = y1 + t * (y2 - y1)
        return (intersection_x, intersection_y)

    return None


def point_line_distance(point, line_start, line_end):

    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    # direction of the line
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy

    if length_sq == 0:  # if start and end is identical
        return np.hypot(px - x1, py - y1)

    # project of the point on the line
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0, min(1, t))  # limits t

    # nearest point to the line
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy

    # distance between point and line
    distance = np.hypot(px - nearest_x, py - nearest_y)

    return distance