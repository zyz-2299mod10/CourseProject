import cv2
import numpy as np
import random
import math
import sys
import os

import argparse ###

# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    
    # Cylindrical projection
    img = cylindrical_projection(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

def read_img_challenge(path):
    img = cv2.imread(path)

    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    new_img[:, :, 0] = cv2.equalizeHist(new_img[:, :, 0])
    new_img = cv2.cvtColor(new_img, cv2.COLOR_YUV2BGR)
    new_img = np.clip(new_img, 5, 254).astype(np.uint8)
    
    new_img = cylindrical_projection(new_img)
    new_img_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    return new_img, new_img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ",img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def creat_im_window(window_name,img):
    cv2.imshow(window_name,img)

# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
class Stitcher:
    def stitch(self, imgs, grays, SIFT_Detector, threshold = 0.75, blend = 'linearBlendingWithConstantWidth', gain = False):
        '''
        most top function
        '''
        # SIFT
        key_points_1, descriptors_1 = SIFT_Detector.detectAndCompute(grays[0], None)
        key_points_2, descriptors_2 = SIFT_Detector.detectAndCompute(grays[1], None)
        
        # Match points
        matches = self.MatchKpts(key_points_1, descriptors_1, key_points_2, descriptors_2, threshold)
        
        # Find best homography with RANSAC
        H = self.H_with_RANSAC(matches, threshold=5, iter_num=2000)

        # Warp
        warp_img = self.warp(imgs[0], imgs[1], H, blend, gain = gain)
        
        return warp_img

    def MatchKpts(self, kpts_1, descriptors_1, kpts_2, descriptor_2, threshold):
        good_matches = []
        for i, feat1 in enumerate(descriptors_1):
            min_idx = -1
            min_dist = np.inf
            sec_idx = -1
            sec_dist = np.inf

            # Lowe’s Ratio test 
            for j, feat2 in enumerate(descriptor_2):
                current_dist = np.linalg.norm(feat1 - feat2)

                if current_dist < min_dist:
                    sec_idx = min_idx
                    sec_dist = min_dist
                    min_idx = j
                    min_dist = current_dist
                elif current_dist < sec_dist and sec_idx != min_idx:
                    sec_idx = j
                    sec_dist = current_dist

            # find good match
            if min_dist <= sec_dist * threshold:
                good_matches.append([
                    (int(kpts_1[i].pt[0]), int(kpts_1[i].pt[1])),
                    (int(kpts_2[min_idx].pt[0]), int(kpts_2[min_idx].pt[1]))
                ])

        return good_matches
    
    def H_with_RANSAC(self, good_matches, threshold = 5, iter_num = 1000):
        img1_kpts = []
        img2_kpts = []
        for kpt1, kpt2 in good_matches:
            img1_kpts.append(list(kpt1))
            img2_kpts.append(list(kpt2))
        img1_kpts = np.array(img1_kpts)
        img2_kpts = np.array(img2_kpts)
        
        max_inliner = 0
        best_H = None        
        for _ in range(iter_num):
            ran_idx = random.sample(range(len(good_matches)), 4)
            H = get_H(img1_kpts[ran_idx], img2_kpts[ran_idx])

            current_inliner = 0            
            for i in range(len(good_matches)):
                if i not in ran_idx:
                    p1 = np.hstack((img1_kpts[i], [1]))
                    p2 = img2_kpts[i]

                    p2_hat = H @ p1.T                    
                    if p2_hat[2] == 0: # avoid divide 0
                        continue                    
                    p2_hat = p2_hat / p2_hat[2]

                    if (np.linalg.norm(p2_hat[:2] - p2) < threshold):
                        current_inliner = current_inliner + 1
            
            if (current_inliner > max_inliner):
                max_inliner = current_inliner
                best_H = H

        return best_H
                
    def warp(self, img1, img2, H, blend, gain = False):
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape

        l_down = np.hstack(([0], [0], [1]))
        l_up = np.hstack(([0], [h1-1], [1]))
        r_down = np.hstack(([w1-1], [0], [1]))
        r_up = np.hstack(([w1-1], [h1-1], [1]))        
        warped_ld = H @ l_down.T
        warped_lu = H @ l_up.T
        warped_rd =  H @ r_down.T
        warped_ru = H @ r_up.T
        x = int(min(min(min(warped_ld[0],warped_lu[0]),
                        min(warped_rd[0], warped_ru[0])),
                        0))
        
        y = int(min(min(min(warped_ld[1],warped_lu[1]),
                        min(warped_rd[1], warped_ru[1])),
                        0))
        size = (w2 + abs(x), h2 + abs(y))

        A = np.float32([[1, 0, -x], [0, 1, -y], [0, 0, 1]])
        warped1 = cv2.warpPerspective(src=img1, M=A@H, dsize=size)
        warped2 = cv2.warpPerspective(src=img2, M=A, dsize=size)

        if gain:
            print('==> calculating gain...')
            img_gain = get_gain([warped1, warped2])
            img_gain = img_gain.mean(axis=1)
            warped1 = (warped1.astype(np.float32) * img_gain[0]).clip(0, 255).astype(np.uint8)
            warped2 = (warped2.astype(np.float32) * img_gain[1]).clip(0, 255).astype(np.uint8)
        
        blender = Blender()
        if blend == 'linearBlendingWithConstantWidth':
            result = blender.LBwCW([warped1, warped2])
        elif blend == 'linearBlending':
            result = blender.LB([warped1, warped2])
        
        return result

class Blender:
    def LB(self, imgs):
        img_left, img_right = imgs
        hl, wl, _ = img_left.shape
        hr, wr, _ = img_right.shape
        img_left_mask = np.zeros((hr, wr), dtype="int")
        img_right_mask = np.zeros((hr, wr), dtype="int")
        
        for i in range(hl):
            for j in range(wl):
                if np.count_nonzero(img_left[i, j]) > 0:
                    img_left_mask[i, j] = 1
        for i in range(hr):
            for j in range(wr):
                if np.count_nonzero(img_right[i, j]) > 0:
                    img_right_mask[i, j] = 1
        
        # find overlap
        overlap_mask = np.zeros((hr, wr), dtype="int")
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                    overlap_mask[i, j] = 1
        
        # compute the alpha mask
        alpha_mask = np.zeros((hr, wr))
        for i in range(hr): 
            minIdx = maxIdx = -1
            for j in range(wr):
                if (overlap_mask[i, j] == 1 and minIdx == -1):
                    minIdx = j
                if (overlap_mask[i, j] == 1):
                    maxIdx = j
            
            if (minIdx == maxIdx):
                continue
                
            decrease_step = 1 / (maxIdx - minIdx)
            for j in range(minIdx, maxIdx + 1):
                alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
        
        linearBlending_img = np.copy(img_right)
        linearBlending_img[:hl, :wl] = np.copy(img_left)
        # linear blending
        for i in range(hr):
            for j in range(wr):
                if ( np.count_nonzero(overlap_mask[i, j]) > 0):
                    linearBlending_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[i, j]
                elif(np.count_nonzero(img_left_mask[i, j]) > 0):
                    linearBlending_img[i, j] = img_left[i, j]
                else:
                    linearBlending_img[i, j] = img_right[i, j]
        return linearBlending_img
    
    def LBwCW(self, imgs, cons_w = 5):
        img_left, img_right = imgs
        hl, wl, _ = img_left.shape
        hr, wr, _ = img_right.shape
        left_mask = np.zeros((hr, wr), dtype="int")
        right_mask = np.zeros((hr, wr), dtype="int")
        
        for i in range(hl):
            for j in range(wl):
                if np.count_nonzero(img_left[i, j]) > 0:
                    left_mask[i, j] = 1
        for i in range(hr):
            for j in range(wr):
                if np.count_nonzero(img_right[i, j]) > 0:
                    right_mask[i, j] = 1
        
        # find overlap
        overlap_mask = np.zeros((hr, wr), dtype="int")
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(left_mask[i, j]) > 0 and np.count_nonzero(right_mask[i, j]) > 0):
                    overlap_mask[i, j] = 1
        
        # compute the alpha mask
        alpha_mask = np.zeros((hr, wr))
        for i in range(hr):
            min_idx = -1
            max_idx = -1
            for j in range(wr):
                if (overlap_mask[i, j] == 1 and min_idx == -1):
                    min_idx = j
                if (overlap_mask[i, j] == 1):
                    max_idx = j            
            if (min_idx == max_idx):
                continue                
            d_step = 1 / (max_idx - min_idx)            
            mid_idx = int((max_idx + min_idx) / 2)
            
            # left 
            for j in range(min_idx, mid_idx + 1):
                if (j >= mid_idx - cons_w):
                    alpha_mask[i, j] = 1 - (d_step * (j - min_idx))
                else:
                    alpha_mask[i, j] = 1
            # right
            for j in range(mid_idx + 1, max_idx + 1):
                if (j <= mid_idx + cons_w):
                    alpha_mask[i, j] = 1 - (d_step * (j - min_idx))
                else:
                    alpha_mask[i, j] = 0
        
        blend_img = np.copy(img_right)
        blend_img[:hl, :wl] = np.copy(img_left)
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(overlap_mask[i, j]) > 0):
                    blend_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[i, j]
                elif(np.count_nonzero(left_mask[i, j]) > 0):
                    blend_img[i, j] = img_left[i, j]
                else:
                    blend_img[i, j] = img_right[i, j]
        return blend_img

def get_H(kpts_1, kpts_2):
    A = []
    for i in range(len(kpts_1)):
        A.append([kpts_1[i, 0], kpts_1[i, 1], 1, 0, 0, 0, -kpts_1[i, 0] * kpts_2[i, 0], -kpts_1[i, 1] * kpts_2[i, 0], -kpts_2[i, 0]])
        A.append([0, 0, 0, kpts_1[i, 0], kpts_1[i, 1], 1, -kpts_1[i, 0] * kpts_2[i, 1], -kpts_1[i, 1] * kpts_2[i, 1], -kpts_2[i, 1]])

    _, _, v = np.linalg.svd(A)
    H = np.reshape(v[8], (3, 3))
    H = H/H.item(8)       
    return H

def get_gain(imgs, sig_n = 100, sig_g = 0.9):
    '''
    Compute the task-gain compensation
    '''
    img_num = len(imgs)    
    coef = np.zeros((img_num, img_num, 3))
    results = np.zeros((img_num, 3))
    for i in range(img_num-1):
        for j in range(i+1, img_num):
            I_ij, I_ji, N_ij = cal_pair(imgs[i], imgs[j], detect_threshold=6)
            if N_ij == 0:
                continue
            coef[i][i] += N_ij * ((2 * I_ij ** 2 / sig_n ** 2) + (1 / sig_g ** 2))
            coef[i][j] -= (2 / sig_n ** 2) * N_ij * I_ij * I_ji
            coef[j][i] -= (2 / sig_n ** 2) * N_ij * I_ji * I_ij        
            coef[j][j] += N_ij * ((2 * I_ji ** 2 / sig_n ** 2) + (1 / sig_g ** 2))

            results[i] += N_ij / sig_g ** 2
            results[j] += N_ij / sig_g ** 2
            
    gains = np.zeros_like(results)
    coef = coef / 1e8
    results = results / 1e8
    for i in range(coef.shape[2]):
        coefs = coef[:, :, i]
        res = results[:, i]
        gains[:, i] = np.linalg.pinv(coefs) @ res

    return gains

# def get_gain(img1, img2, sig_n = 100, sig_g = 0.9):
#     '''
#     Compute the task-gain compensation
#     '''
#     coef = np.zeros((2, 2, 3))
#     results = np.zeros((2, 3))    
#     I_ij, I_ji, N_ij = cal_pair(img1, img2, detect_threshold=6)
#     if N_ij == 0:
#         return
#     coef[0][0] += N_ij * ((2 * I_ij ** 2 / sig_n ** 2) + (1 / sig_g ** 2))
#     coef[0][1] -= (2 / sig_n ** 2) * N_ij * I_ij * I_ji
#     coef[1][0] -= (2 / sig_n ** 2) * N_ij * I_ji * I_ij        
#     coef[1][1] += N_ij * ((2 * I_ji ** 2 / sig_n ** 2) + (1 / sig_g ** 2))

#     results[0] += N_ij / sig_g ** 2
#     results[1] += N_ij / sig_g ** 2
            
            
#     gains = np.zeros_like(results)
#     coef = coef / 1e8
#     results = results / 1e8
#     for i in range(coef.shape[2]):
#         coefs = coef[:, :, i]
#         res = results[:, i]
#         gains[:, i] = np.linalg.pinv(coefs) @ res

#     return gains

def cal_pair(img1, img2, detect_threshold = 6):
    width, height, _ = img1.shape   

    img1_mask = np.zeros((height, width), dtype=np.int16)
    img2_mask = np.zeros((height, width), dtype=np.int16)
    for i in range(height):
        for j in range(width):
            if np.sum(img1[i, j]) > detect_threshold:
                img1_mask[i, j] = 1
            if np.sum(img2[i, j]) > detect_threshold:
                img2_mask[i, j] = 1
                  
    overlap_mask = img1_mask * img2_mask
    N_ij = np.count_nonzero(overlap_mask)
    if N_ij == 0:
        return None, None, 0
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    overlap_mask = overlap_mask.astype(np.float32)
    I_ij = np.sum(img1 * np.stack([overlap_mask, overlap_mask, overlap_mask], axis=2), axis=(0, 1)) / N_ij
    I_ji = np.sum(img2 * np.stack([overlap_mask, overlap_mask, overlap_mask], axis=2), axis=(0, 1)) / N_ij
    return I_ij, I_ji, N_ij

def compute_camera_intrinsics_matrix(image_heigth, image_width, horizontal_fov):
    '''
    calculate intrinsic matrix
    '''
    vertical_fov = (image_heigth / image_width * horizontal_fov) * np.pi / 180
    horizontal_fov *= np.pi / 180

    f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
    f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)

    K = np.array([[f_x, 0.0, image_width / 2.0], [0.0, f_y, image_heigth / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K

def cylindrical_projection(img):    
    height, width, _ = img.shape
    K = compute_camera_intrinsics_matrix(height, width, 55) # intrinsic matrix
    foc_len = (K[0][0] + K[1][1]) / 2

    cylinder = np.zeros_like(img)
    temp = np.mgrid[0:width, 0:height]
    x, y = temp[0], temp[1]

    # Compute theta and height
    theta = (x - K[0][2]) / foc_len  # angle theta
    h = (y - K[1][2]) / foc_len  # height

    # Create points in cylindrical coordinate
    p = np.array([np.sin(theta), h, np.cos(theta)])
    p = p.T
    p = p.reshape(-1, 3)

    # Project points
    image_points = K.dot(p.T).T

    # Normalize
    points = image_points[:, :-1] / image_points[:, [-1]]
    points = points.reshape(height, width, -1)

    cylinder = cv2.remap(img, points[:, :, 0].astype(np.float32), points[:, :, 1].astype(np.float32), cv2.INTER_LINEAR)
    return cylinder

def Base():
    photo_path = './Photos'

    SIFT_Detector = cv2.SIFT_create()
    stitcher = Stitcher()
    blend = None
    
    mode = 'Base'
    img_name = ['Base1.jpg', 'Base2.jpg', 'Base3.jpg']
    # blend = 'linearBlending'
    blend = 'linearBlendingWithConstantWidth'

    img1, img_gray1 = read_img(os.path.join(photo_path, mode, img_name[0]))
    img2, img_gray2 = read_img(os.path.join(photo_path, mode, img_name[1]))

    result_img = stitcher.stitch([img1, img2], [img_gray1, img_gray2], SIFT_Detector, threshold = 0.75, blend = blend, gain = False)
    result_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(photo_path, mode, 'First_2_result.jpg'), result_img)

    for idx, img_name in enumerate(img_name[2:], start = 3):
        next_img, next_img_gray = read_img(os.path.join(photo_path, mode, img_name))
        result_img = stitcher.stitch([result_img, next_img], [result_gray, next_img_gray], SIFT_Detector, threshold = 0.75, blend = blend, gain = False)
        result_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)                
        if idx != len(img_name): 
            cv2.imwrite(os.path.join(photo_path, mode, f'First_{idx}_result.jpg'), result_img)
            
    cv2.imwrite(os.path.join(photo_path, mode, 'Final_result.jpg'), result_img)    

def Challenge():
    photo_path = './Photos'

    SIFT_Detector = cv2.SIFT_create()
    stitcher = Stitcher()
    blend = None

    mode = 'Challenge'
    img_name = ['Challenge1.jpg', 'Challenge2.jpg', 'Challenge3.jpg', 'Challenge4.jpg', 'Challenge5.jpg', 'Challenge6.jpg']
    blend = 'linearBlendingWithConstantWidth'

    img1, img_gray1 = read_img_challenge(os.path.join(photo_path, mode, img_name[0]))
    img2, img_gray2 = read_img_challenge(os.path.join(photo_path, mode, img_name[1]))

    result_img = stitcher.stitch([img1, img2], [img_gray1, img_gray2], SIFT_Detector, threshold = 0.75, blend = blend, gain = True)
    result_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)            
    cv2.imwrite(os.path.join(photo_path, mode, 'First_2_result.jpg'), result_img)

    for idx, img_name in enumerate(img_name[2:], start = 3):
        next_img, next_img_gray = read_img_challenge(os.path.join(photo_path, mode, img_name))                
        result_img = stitcher.stitch([result_img, next_img], [result_gray, next_img_gray], SIFT_Detector, threshold = 0.75, blend = blend, gain = True)
        result_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)                
        if idx != len(img_name):
            cv2.imwrite(os.path.join(photo_path, mode, f'First_{idx}_result.jpg'), result_img)
            
    cv2.imwrite(os.path.join(photo_path, mode, 'Final_result.jpg'), result_img)


def get_args(): ###
    opt = argparse.ArgumentParser()
    opt.add_argument("--mode",
                     type = str,
                     choices=['Base', 'Challenge', 'all'],
                     default='all')
    
    args = opt.parse_args()
    return args

if __name__ == '__main__':
    # the example of image window
    # creat_im_window("Result",img)
    # im_show()

    args = get_args()
    if args.mode == 'Base': ###
        Base()    
    elif args.mode == 'Challenge':
        Challenge()
    elif args.mode == 'all':
        Base()
        Challenge()

    # you can use this function to store the result
    # cv2.imwrite("result.jpg",img)