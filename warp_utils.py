import cv2
import numpy as np


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


if __name__ == "__main__":
    path_to_img1 = "/home/hakito/python_scripts/AirSim/RGB/1722166305487264768.png"
    path_to_img2 = "/home/hakito/python_scripts/AirSim/RGB/1722166305626541568.png"

    im1 = cv2.imread(path_to_img1, cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(path_to_img2, cv2.IMREAD_GRAYSCALE)
    flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    print(flow.dtype)

    hsv = draw_hsv(flow)
    im2w = warp_flow(im1, flow)

    im1_f = cv2.convertScaleAbs(im1, alpha=1.0, beta=0.0).astype(float)
    im2_f = cv2.convertScaleAbs(im2, alpha=1.0, beta=0.0).astype(float)
    im2w_f = cv2.convertScaleAbs(im2w, alpha=1.0, beta=0.0).astype(float)

    err = np.abs(im2_f - im2w_f)

    print(err.min())
    print(err.max())
    print(err.mean())

    cv2.imwrite("temp/flow.png",hsv)
    cv2.imwrite("temp/im1.png", im1)
    cv2.imwrite("temp/im2.png", im2)
    cv2.imwrite("temp/im2w.png", im2w)
    cv2.imwrite("temp/im2w.png", im2w)
    cv2.imwrite("temp/err.png", err)