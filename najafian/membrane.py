""" Module that handles processing the membrane edge mask to produce useful linearized contours for the slits.py module """
import cv2
import numpy as np
from skimage.morphology import skeletonize

from najafian.branches import *
from najafian.util import *

# create the membrane properties
# the min threshold (0-255) for the membrane to be considered valid
MIN_MEMBRANE_THRESHOLD = 25
MIN_MEMBRANE_LENGTH = 40  # the min membrane length
MAX_CORNER_EDGE_DISTANCE = 10  # distance for a corner to be the edge of the image
MAX_CORNER_DISTANCE = 150  # max distance that corners can be from each other
EPSILON_MEMBRANE = 1  # amount to simplify the polygon


def process_membrane(layer):
    """ This function has quite a few steps associated with it to process the membrane mask

    Steps:
      1. threshold the mask layer
      2. skeletonize all the membrane masks into a single pixel line
      3. using the branches.py module and mahotas find all of the branches in the membrane
             a branch is where one line splits into two or more lines
      4. at each branch point draw a small black circle to delete that branch
      5. isolate each membrane contour and apply the Shi-Tomasi Corner Detector to find the ends of the membrane (dilating slowly until a good match is found)
      6. create the corner lines which defines the linearized version (from one edge of contour to the other) of the skeletonized line
      7. membrane joining method is applied
           a. iterate through each membrane segment (with its ends) and compare it to all other segments distances
           b. if the ends are close enough and it's not in the connected corners list try to merge the two membrane segments
           c. take the first segment and if the second segments last point is closest to this last point then reverse the second segment to create one segment
           d. the same step is applied to other scenarios such as first-last, last-last, first-first, and last-first (easiest)
           e. simplify the linearized segment using the provided epsilon above 
      8. partially as safety method and a revision method all connected membranes are redrawn on an overlapping image
            where it can be treated as an OR if two segments are connected the same way but reversed then the same revised line will be resulted (no duplicates)
      9. finally, the same corners, as used in #5, are used to linearized the found contours in the overlapping image

    Args:
        layer (np.ndarray): membrane edge layer image (mask that hasn't been threshed)

    Returns:
        tuple: revised_lines (a list of np.ndarray points), contours (a list of np.ndarray contours), lengths (list of float the lengths of each line)
    """
    # threshold mask
    threshed = cv2.threshold(
        layer, MIN_MEMBRANE_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    # skeletonize
    skeleton = skeletonize(threshed / 255)
    branch_points = detect_branch_points(skeleton)
    skeleton = skeleton.astype(np.uint8) * 255

    # destroy the branch connections
    for x, y in branch_points:
        cv2.rectangle(skeleton, (x - 3, y - 3), (x + 3, y + 3), 0, -1)

    # subdivide each skeleton contour and find the best two edges of each contour
    # this will indicate how we can linearize the contour
    contours = cv2.findContours(
        skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    height, width = layer.shape[:2]
    left, top, right, bottom = 0, 0, width - 1, height - 1
    corner_lines = []
    padding = 10
    for cont in contours:
        # redraw the contour with padding on a new image called single_contour
        x, y, w, h = cv2.boundingRect(cont)
        single_contour = np.zeros(
            (h + 2 * padding, w + 2 * padding, 1), np.uint8)
        offset = np.array([(int(p_x - x + padding), int(p_y - y + padding)) for p_x, p_y in cont.reshape(-1, 2)],
                          np.int32)
        cv2.drawContours(single_contour, [offset], 0, 255, 1)
        edges = cv2.dilate(single_contour, np.ones(
            (5, 5), np.uint8), iterations=1)

        test_ind = 0
        corners = []
        while test_ind < 20:  # test a max of 20 branched
            passed = True
            corns = cv2.goodFeaturesToTrack(edges, 2 + test_ind, 0.3, 20)

            corners = []
            if corns is not None:
                failed = 0
                for corner in corns:
                    p_x, p_y = corner.ravel()
                    p_x, p_y = int(p_x + x - padding), int(p_y + y - padding)
                    corners.append((p_x, p_y))
                if failed == test_ind + 1:
                    passed = False

            if passed:
                break

            test_ind += 1

        # we need to have two corners to continue
        if len(corners) < 2:
            continue

        vectors = cont.reshape(-1, 2)
        first = closest_to(corners[0], vectors)
        second = closest_to(corners[1], vectors)
        start = min(first, second)
        end = max(first, second)
        corner_lines.append(
            (vectors[start:end], (vectors[start], vectors[end])))

    # revisit the lines with corners and try to attach nearby contours
    overlap_lines = []
    ind_line = 0
    connected_lines = []
    connected_corners = []
    for line, corners in corner_lines:
        c_line = list(line.reshape(-1, 2))
        for corner in corners:
            closest = sys.maxsize
            closest_point = None
            is_end = False

            # find other segments to see the closest one
            for o_line, o_corners in corner_lines[ind_line + 1:]:
                for o_corner in o_corners:
                    dist = point_distance(corner, o_corner)
                    x, y = tuple(corner)
                    if x - MAX_CORNER_EDGE_DISTANCE <= left or x + MAX_CORNER_EDGE_DISTANCE > right or \
                            y - MAX_CORNER_EDGE_DISTANCE <= top or y + MAX_CORNER_EDGE_DISTANCE >= bottom:
                        continue
                    elif dist < closest and dist < MAX_CORNER_DISTANCE:
                        closest = dist
                        closest_point = (o_line, o_corner)
                        is_end = bool(tuple(o_corner) == tuple(o_corners[-1]))

            # combine the segments if not already combined
            if closest_point is not None and \
                    not (tuple(corner) in connected_corners or tuple(closest_point[1]) in connected_corners):
                s_line = list(closest_point[0].reshape(-1, 2))
                is_first_end = bool(tuple(corner) == tuple(corners[-1]))

                # order the lines properly
                if is_first_end and not is_end:
                    c_line.extend(s_line)
                elif is_first_end and is_end:
                    c_line.extend(list(reversed(s_line)))
                elif not is_first_end and not is_end:
                    c_line = list(reversed(c_line))
                    c_line.extend(s_line)
                else:
                    temp = c_line[:]
                    c_line = s_line
                    c_line.extend(temp)

                # add it to the already connected corners and lines
                connected_corners.extend(
                    [tuple(corner), tuple(closest_point[1])])
                connected_lines.append((c_line, closest_point[0]))

        # process the revised line to make sure it's up to spec
        # we want to simplify the line
        line = np.array(c_line, np.int32)
        length = cv2.arcLength(line, False)
        if length >= MIN_MEMBRANE_LENGTH:
            smooth_contours = cv2.approxPolyDP(
                line, EPSILON_MEMBRANE, False).reshape(-1, 2)
            overlap_lines.append(smooth_contours)
        ind_line += 1

    # try to find corners that have not been detected
    overlap_image = np.zeros((height, width), np.uint8)

    # draw the connected overlapping lines
    for line in overlap_lines:
        draw_lines(overlap_image, line, 255, 1, False)

    # draw small circles at the connected corners to smooth out the line
    for corner in connected_corners:
        cv2.circle(overlap_image, corner, 1, 255, -1)

    # determine the none connected corners
    nonconnected_corners = []
    for _, corners in corner_lines:
        for corner in corners:
            if tuple(corner) not in connected_corners:
                nonconnected_corners.append(tuple(corner))

    # revise the lines to regenerate contours and the connected lines
    contours = cv2.findContours(
        overlap_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    revised_lines = []
    lengths = []
    for cont in contours:
        vectors = cont.reshape(-1, 2)
        closest_corners = []

        # apply for both corners (test done with not in closest)
        for _ in range(2):
            closest_corner = (0, 0)
            closest_length = sys.maxsize
            for corner in nonconnected_corners:
                if corner not in closest_corners:
                    length = point_distance(
                        corner, vectors[closest_to(corner, vectors)])
                    if length < closest_length:
                        closest_length = length
                        closest_corner = corner
            closest_corners.append(closest_corner)

        first = closest_to(closest_corners[0], vectors)
        second = closest_to(closest_corners[1], vectors)
        start = min(first, second)
        end = max(first, second)
        line = vectors[start:end]
        revised_lines.append(line)
        lengths.append(cv2.arcLength(line, False))

    return revised_lines, contours, lengths
