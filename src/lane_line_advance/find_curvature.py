import cv2
import numpy as np


def birds_eye_view(img, src_points, dst_points):
    """
    :param img:
    :param src_points: list(tuples)) or nd_array (4, 2)
    :param dst_points:
    :return:
    """
    h, w, _ = img.shape
    src_points = np.array(src_points).astype(np.float32)
    dst_points = np.array(dst_points).astype(np.float32)
    
    assert(src_points.shape[1] == 2 and src_points.shape[0] >= 4)
    assert(dst_points.shape[1] == 2 and dst_points.shape[0] >= 4)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_NEAREST)
    return warped_image
