# use the provided landscape to create panorama.
# use the Poisson blending.

import cv2
import numpy as np


def get_corners(image):
    corners = np.zeros((4, 1, 2), dtype=np.float32)
    shape = image.shape
    y = shape[0]
    x = shape[1]
    corners[0] = [0, 0]
    corners[1] = [0, y]
    corners[2] = [x, y]
    corners[3] = [x, 0]
    return corners

def compute_min(result_dims, axis):
    if axis == 'x':
        x_min = int(result_dims.min(axis=0).reshape(-1, order='A')[0])
        return x_min
    if axis == 'y':
        y_min = int(result_dims.min(axis=0).reshape(-1, order='A')[1])
        return y_min

def compute_max(result_dims, axis):
    if axis == 'x':
        x_max = int(result_dims.max(axis=0).reshape(-1, order='A')[0])
        return x_max
    if axis == 'y':
        y_max = int(result_dims.max(axis=0).reshape(-1, order='A')[1])
        return y_max

def compute_homo(img1, img2, threshold):
    sift = cv2.xfeatures2d.SIFT_create()
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    key_points, desc = sift.detectAndCompute(gray, None)
    key_points2, desc2 = sift.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc, desc2, k=2)
    match_array = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            match_array.append(m)
    img1_pts = []
    img2_pts = []
    for match in match_array:
        img1_pts.append(key_points[match.queryIdx].pt)
        img2_pts.append(key_points2[match.trainIdx].pt)
    img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
    img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
    return M


def blend_image(img1, img2, M):
    dim1 = get_corners(img1).reshape(-1, 1, 2)
    dim2_temp = get_corners(img2).reshape(-1, 1, 2)
    dim2 = cv2.perspectiveTransform(dim2_temp, M)
    result_dims = np.concatenate((dim1, dim2))
    x_min = compute_min(result_dims, 'x')
    y_min = compute_min(result_dims, 'y')
    x_max = compute_max(result_dims, 'x')
    y_max = compute_max(result_dims, 'y')
    t_array = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    result = cv2.warpPerspective(img2, t_array.dot(M), (x_max - x_min, y_max - y_min))
    result[-y_min: img1.shape[:2][0] + -y_min, -x_min: img1.shape[:2][1] + -x_min] = img1
    return result


img1 = cv2.imread('landscape_1.jpg')
img2 = cv2.imread('landscape_2.jpg')
img3 = cv2.imread('landscape_3.jpg')
img4 = cv2.imread('landscape_4.jpg')
img5 = cv2.imread('landscape_5.jpg')
img6 = cv2.imread('landscape_6.jpg')
img7 = cv2.imread('landscape_7.jpg')
img8 = cv2.imread('landscape_8.jpg')
img9 = cv2.imread('landscape_9.jpg')

M_1 = compute_homo(img1, img2, 0.4)
result_1 = blend_image(img2, img1, M_1)
M_2 = compute_homo(img3, img4, 0.4)
result_2 = blend_image(img4, img3, M_2)
M_3 = compute_homo(img7, img6, 0.4)
result_3 = blend_image(img6, img7, M_3)
M_4 = compute_homo(img9, img8, 0.4)
result_4 = blend_image(img8, img9, M_4)

M_5 = compute_homo(result_1, result_2, 0.4)
result_5 = blend_image(result_2, result_1, M_5)
M_6 = compute_homo(result_4, result_3, 0.4)
result_6 = blend_image(result_3, result_4, M_6)

M_7 = compute_homo(result_5, img5, 0.4)
result_7 = blend_image(img5, result_5, M_7)

M_8 = compute_homo(result_7, result_6, 0.4)
result_8 = blend_image(result_6, result_7, M_8)

cv2.imwrite('panorama_8.jpg', result_8)
