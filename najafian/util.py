""" Contains many useful functions, some maybe unused, for common math operations in slits.py """
import sys
import cv2
import numpy as np


def point_distance(pt1, pt2):
    """ Gets the distance between two points

    Args:
        pt1 (tuple): x, y
        pt2 (tuple): x, y

    Returns:
        float: distance between points
    """
    return (((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2)) ** 0.5


def point_slope(pt1, pt2):
    """ Gets the slope from pt1 to pt2

    Args:
        pt1 (tuple): x, y
        pt2 (tuple): x, y

    Returns:
        float: slope between points (will return sys.maxsize if pt1.x == pt2.x)
    """
    if pt1[0] == pt2[0]:
        return sys.maxsize
    return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])


def point_rad(pt1, pt2):
    """ Gets the angle in radians between pt1 and pt2

    Args:
        pt1 (tuple): x, y
        pt2 (tuple): x, y

    Returns:
        float: angle between points from 0 reference point (remember zero in disp is top-left)
    """
    if pt2[0] < pt1[0]:
        temp = pt1
        pt1 = pt2
        pt2 = temp
    return np.arctan(point_slope(pt1, pt2))


def point_center(pt1, pt2, type=int):
    """ Get the center point between pt1 and pt2

    Args:
        pt1 (tuple): x, y
        pt2 (tuple): x, y
        type ([type], optional): object conversion (ex float/int). Defaults to int.

    Returns:
        tuple: a new center point of ([type], [type]) 
    """
    return type((pt1[0] + pt2[0]) / 2), type((pt1[1] + pt2[1]) / 2)


def is_line_in_edge(cont, pt1, pt2, checks=3):
    """ Determines if the line from pt1 and pt2 remains in the edge/inside a polygon of cont

    Note: this is a simplified method that will divide the line into segments and check
        some points within the line and the contour. So when checks increases the more 
        time it takes to processes, but the more accurate this method will be

    Args:
        cont (np.ndarray): (N, 2) contour
        pt1 (tuple): x, y
        pt2 (tuple): x, y
        checks (int, optional): how many times to subdivide the line to check. Defaults to 3.

    Returns:
        [type]: [description]
    """
    slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])

    def line_to(perc):
        x = pt1[0] + (perc * (pt2[0] - pt1[0]))
        return x, slope * (x - pt1[0]) + pt1[1]

    # check if it's inside or on the edge at multiple points
    check_size = 1.0 / float(checks)
    for i in range(checks):
        if cv2.pointPolygonTest(cont, line_to(check_size * i), False) < 0:
            return False

    # all points are inside the contour
    return True


def subdivide(contours, times=1, dtype=np.float32):
    """ Subdivides a contour by getting the center between lines and adding it to the contour. This is useful
    for simple distance/closest point measuring without having to do any extra math/processing

    Args:
        contours (np.ndarray): (N, 2) contour
        times (int, optional): Number of times to subdivide the contour (normally this doubles the size of the contour each subdivide). Defaults to 1.
        dtype (np.dtype, optional): The resulting contour type. Defaults to np.float32.
    """
    # this will iterate through each pair of points and append the center point between the two
    def sub(cont):
        sub_ind = 0
        if len(cont) > 1:
            sub_cont = np.zeros(((cont.shape[0] * 2) - 1, 2), dtype=dtype)
            for i in range(len(cont) - 1):
                sub_cont[sub_ind] = cont[i]
                sub_ind += 1  # shift the index from center
                sub_cont[sub_ind] = point_center(cont[i], cont[i + 1], type=float if dtype == np.float32 else int)
                sub_ind += 1  # new point added so shift to next point
            sub_cont[sub_ind] = cont[-1]
            return sub_cont
        return cont

    # subdivide the contour N times
    sub_cont = contours.copy()
    for _ in range(times):
        sub_cont = sub(sub_cont)

    return sub_cont


def closest_to(pt, cont):
    """ Gets the closest index of contour "cont" to the point "pt" 

    Args:
        pt (tuple): x, y point
        cont (np.ndarray): (N, 2) contour

    Returns:
        int: index in cont that the point pt is closest to
    """
    deltas = cont - pt
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


def center_mass(cont):
    """ Gets the center of the moment of a contour 

    Args:
        cont (np.ndarray): (N, 2) contour

    Returns:
        tuple: x, y point of the center of mass
    """
    M = cv2.moments(cont)
    return int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])


def draw_lines(img, series, color, thickness, circles=False):
    """ Draws a series of lines or circles with the specified thickness and color

    Args:
        img (np.ndarray): image to draw on
        series (list of points or np.ndarray): draws the specified lines (can have rounded edges) 
        color (obj): an opencv acceptable color
        thickness (int): thickness of lines/circles [Note: -1 does not work (all circles are filled)] 
        circles (bool, optional): draw list of circles instead of lines. Defaults to False.
    """
    if len(series) > 0 and circles:
        cv2.circle(img, tuple(series[0]), 4, color, -1)

    for i in range(len(series) - 1):
        cv2.line(img, tuple(series[i]), tuple(series[i + 1]), color, thickness)

        if circles:
            cv2.circle(img, tuple(series[i + 1]), 4, color, -1)
