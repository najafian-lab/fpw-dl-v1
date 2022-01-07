""" Module handles identifying slits in the slit mask and then measuring the distances between slits from the linearized contours provided by membrane.py """
import random
import sys

import cv2
import numpy as np

from najafian.util import *

# --- ADJUST PARAMS ---
# slit identification
SLIT_MIN_AREA = 3   # min area in pixels for a slit to be
SLIT_MAX_AREA = 100  # max area in pixels for a slit to be
SLIT_MIN_THRESHOLD = 55   # min mask threshold for slit mask
SLIT_MIN_DISTANCE_BETWEEN_BLOBS = 1  # minimum distance between blobs

# slit to membrane correlation properties
MAX_MEMBRANE_SLIT_DISTANCE = 15  # max distance the slits can be from the membrane
CERTAIN_MEMBRANE_SLIT_DISTANCE = 3 # distance from membrane to immediately break out of the loop
# -- END ADJUST PARAMS --


# keep track of biopsy statistics and obj identification
pp = cv2.SimpleBlobDetector_Params()
pp.filterByArea = True
pp.minArea = SLIT_MIN_AREA
pp.maxArea = SLIT_MAX_AREA
pp.filterByCircularity = False
pp.minCircularity = 0.05
pp.maxCircularity = 1.0
pp.minThreshold = SLIT_MIN_THRESHOLD
pp.minDistBetweenBlobs = SLIT_MIN_DISTANCE_BETWEEN_BLOBS
pp.filterByColor = False
pp.filterByConvexity = False
pp.filterByInertia = False

# create the blob detector based off of the above parameters
blober = cv2.SimpleBlobDetector_create(pp)


def process_slits(layer, contours, lines, lengths, ilog=None, ret_more=False):
    """ A series of steps applied to a slit mask to measure the distance between slits on a membrane edge

    Steps:
      1. Using the cv2 blob detector to detect the slits inside the mask
      2. for each slit test to see if the slit intersects a line and add it to the slit lines
           if it's not then iterate through each line and find the closest line to append it to
      3. since the slits are not properly ordered to each line another loop is applied
           this time iterating through each point on each membrane and finding the closest
           slit in the membrane group to that point, if it's within our certain distance it assigns it there,
           if it's not then it scans the rest of the line to see if another point is closer
      4. Finally for each membrane segment (grouped with its ordered slits) the following algorithm is applied
          a. We subdivide the membrane points 3 times
          b. We iterate through each slit pair (a-b, b-c, c-d, etc) and get the closest point of a to the membrane and b to the membrane
                and call them the start/end indices
          c. with each start/end we do an cv2.arcLength on the linearized membrane segment to get its distance and add it to our global distance list

    Args:
        layer (np.ndarray): slit mask layer
        contours (list of np.ndarray): list of (N, 2) contours for the membrane skeletons
        lines (list of np.ndarray): same as contours except the linearized versions from the ends of the membrane
        lengths (list of float): the total length of each membrane line 
        ilog (method or None, optional): method to log a preview image, else None means no preview. Defaults to None
        ret_more (bool, optional): [description]. Defaults to False.

    Returns:
        tuple: distances (list of float), avg_distance (float), slit_count (int), attachments (list of float), actual_slit_locations (list of tuple (x, y)) [only if ret_more is True]
    """
    keypoints = blober.detect(layer)
    lines_range = range(len(lines))
    slit_lines = [[] for _ in lines_range]

    slit_count = 0
    actual_slit_locations = []
    for kp in keypoints:
        closest = -1
        distance = sys.maxsize
        for i in lines_range:
            dist = -cv2.pointPolygonTest(lines[i], kp.pt, True)

            # if it's inside the line then quite
            if dist <= 0:
                closest = i
                break
                # if it's close to the line let's continue to see if we can find a closer one
            elif dist < distance:
                distance = dist
                closest = i

        # we didn't find a close line
        if closest == -1:
            continue

        # add to the slit count
        slit_count += 1

        # add the slit keypoint to the line list
        slit_lines[closest].append(tuple(int(i) for i in kp.pt))
        actual_slit_locations.append(tuple(int(i) for i in kp.pt))

    # print(len(slit_lines[0]))
    img = np.zeros(layer.shape[:2] + (3,), np.uint8)

    # create the new ordered line list
    ordered_lines = []
    line_ind = -1
    for line in slit_lines:
        line_ind += 1

        # we can't compute the distance between zero or one slits
        if len(line) < 2:
            ordered_lines.append([])
            continue

        # create the ordered from the random line points
        shift_lines = []

        # shift through the entire line contour
        for pt_ind, pt in enumerate(lines[line_ind]):
            closest = -1
            distance = sys.maxsize
            ind = 0
            while ind < len(keypoints):
                # get the distance of that point to this line
                dist = point_distance(pt, keypoints[ind].pt)

                # just select this point because it's practically on the line
                if dist < CERTAIN_MEMBRANE_SLIT_DISTANCE:
                    closest = ind
                    break
                    # let's not check loopbacks unless it meets or max dist requirements
                elif dist < MAX_MEMBRANE_SLIT_DISTANCE and dist < distance:
                    # shift through the rest of the line to make sure the point isn't closer to something else
                    # this could occur if the line loops back around and we don't know about it yet
                    passed = True
                    for o_pt in lines[line_ind][pt_ind + 1:]:
                        o_dist = point_distance(o_pt, keypoints[ind].pt)
                        if o_dist < dist:
                            passed = False
                            break

                    if passed:
                        distance = dist
                        closest = ind
                ind += 1

            # we didn't find a valid point
            if closest == -1:
                continue

            shift_lines.append(keypoints[closest].pt)
            del keypoints[closest]
        ordered_lines.append(shift_lines)

    # convert all of the lines
    slit_lines = [np.array(line, np.int32) for line in ordered_lines]
    # print(slit_lines[0].shape)

    # show a preview (if enabled)
    if ilog is not None:
        # cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
        for ind, line in enumerate(lines):
            draw_lines(img, line, (255, 0, 0), 1, False)

            if line.reshape(-1, 2).shape[0] >= 2:
                x, y = point_center(line[0], line[-1])
                cv2.putText(img, '%.2f' % lengths[ind], (x + 20, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

        for line in slit_lines:
            color = tuple(random.randint(50, 255) for _ in range(3))
            draw_lines(img, line, color, 1, True)

    # calculate the surface distance
    distances = []
    attachments = []
    for line_ind, line in enumerate(slit_lines):
        # get the origin membrane layer and subdivide the contour a few times
        cont = subdivide(lines[line_ind], 3)
        sub_dist = []
        for i in range(len(line) - 1):
            color = (0, 0, 255)

            # get the closet points to the membrane from the slits
            start = closest_to(cont, line[i])
            end = closest_to(cont, line[i + 1])

            # load the subdivision
            segment = cont[start:end + 1]

            # append the surface distance
            distance = cv2.arcLength(segment, False)
            sub_dist.append(distance)
            distances.append(distance)

            # draw the outline
            if ilog is not None and len(segment) >= 2:
                # draw the distance results
                x, y = point_center(segment[0], segment[-1])
                cv2.putText(img, '%.2f' % distance, (x + 20, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                draw_lines(img, segment, color, 3)

        attachments.append(np.sum(sub_dist))

    # calculate the average distance
    if len(distances) == 0:
        avg_distance = -1
    else:
        avg_distance = np.mean(distances, dtype=np.float32)

    if ilog is not None:
        cv2.putText(img, 'Surface Length: %.2f' % float(np.sum(lengths)),
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 55, 255), 1)
        cv2.putText(img, 'Avg Length: %.2f' % avg_distance, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (55, 255, 255), 1)
        ilog(img, delay=1)

    if ret_more:
        return distances, avg_distance, slit_count, attachments, actual_slit_locations
    return distances, avg_distance, slit_count, attachments
