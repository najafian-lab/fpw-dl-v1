import numpy as np
from mahotas.morph import hitmiss as hit_or_miss


def detect_branch_points(input):
    def hit_miss(image, kernel):
        return hit_or_miss(image, np.array(kernel))

    def multi_hit_miss(image, *kernels):
        blank = np.zeros(image.shape, np.int)
        for kernel in kernels:
            blank = np.bitwise_or(blank, hit_miss(image, kernel))
        return blank

    # X, T and Y kernels
    hits_misses = multi_hit_miss(input,
                                 [[0, 1, 0],
                                  [1, 1, 1],  # X
                                  [0, 1, 0]],

                                 [[1, 0, 1],
                                  [0, 1, 0],  # Rotated X
                                  [1, 0, 1]],

                                 [[2, 1, 2],
                                  [1, 1, 1],  # T
                                  [2, 2, 2]],

                                 [[1, 2, 1],
                                  [2, 1, 2],
                                  [1, 2, 2]],

                                 [[2, 1, 2],
                                  [1, 1, 2],
                                  [2, 1, 2]],

                                 [[1, 2, 2],
                                  [2, 1, 2],
                                  [1, 2, 1]],

                                 [[2, 2, 2],
                                  [1, 1, 1],
                                  [2, 1, 2]],

                                 [[2, 2, 1],
                                  [2, 1, 2],
                                  [1, 2, 1]],

                                 [[2, 1, 2],
                                  [2, 1, 1],
                                  [2, 1, 2]],

                                 [[1, 2, 1],
                                  [2, 1, 2],
                                  [2, 2, 1]],

                                 [[1, 0, 1],
                                  [0, 1, 0],  # Y
                                  [2, 1, 2]],

                                 [[0, 1, 0],
                                  [1, 1, 2],
                                  [0, 2, 1]],

                                 [[1, 0, 2],
                                  [0, 1, 1],
                                  [1, 0, 2]],

                                 [[1, 0, 2],
                                  [0, 1, 1],
                                  [1, 0, 2]],

                                 [[0, 2, 1],
                                  [1, 1, 2],
                                  [0, 1, 0]],

                                 [[2, 1, 2],
                                  [0, 1, 0],
                                  [1, 0, 1]],

                                 [[1, 2, 0],
                                  [2, 1, 1],
                                  [0, 1, 0]],

                                 [[2, 0, 1],
                                  [1, 1, 0],
                                  [2, 0, 1]],

                                 [[0, 1, 0],
                                  [2, 1, 1],
                                  [1, 2, 0]])
    hits_misses = np.clip(hits_misses, 0, 1).astype(np.uint8) * 255
    y_hits, x_hits = np.nonzero(hits_misses)
    return [(x_hits[i], y_hits[i]) for i in range(len(x_hits))]
