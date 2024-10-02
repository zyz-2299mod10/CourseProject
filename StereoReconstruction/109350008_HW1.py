import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
import re

image_row = 120
image_col = 120

# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z,filepath):
    Z_map = np.reshape(Z, (image_row,image_col)).copy()
    data = np.zeros((image_row*image_col,3),dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

# read the .bmp file
def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    image_row , image_col = image.shape
    return image

def gaussian_filter(img):
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64)
    kernel /= np.sum(kernel)  # Normalize the kernel

    pad_image = np.pad(img, 1, mode='edge').astype(np.float64)
    gau_image = np.zeros_like(img, dtype=np.float64)

    for i in range(1, pad_image.shape[0]-1):
        for j in range(1, pad_image.shape[1]-1):
            gau_image[i-1, j-1] = np.sum(pad_image[i-1:i+2, j-1:j+2] * kernel)

    return gau_image

def low_pass_filter(img):
    kernel = np.ones((3, 3), dtype=np.float32) / 9
    pad_image = np.pad(img, 1, mode='edge').astype(np.float32)
    low_pass_image = np.zeros_like(img, dtype=np.float32)

    for i in range(1, pad_image.shape[0]-1):
        for j in range(1, pad_image.shape[1]-1):
            low_pass_image[i-1, j-1] = np.sum(pad_image[i-1:i+2, j-1:j+2] * kernel)

    return low_pass_image

def high_pass_filter(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
    pad_image = np.pad(img, 1, mode='edge').astype(np.float32)
    high_pass_image = np.zeros_like(img, dtype=np.float32)

    for i in range(1, pad_image.shape[0]-1):
        for j in range(1, pad_image.shape[1]-1):
            high_pass_image[i-1, j-1] = np.sum(pad_image[i-1:i+2, j-1:j+2] * kernel)

    return high_pass_image

def calculate_normal(I, L):
    # KdN = (LT。L)-1 。 LT 。 I
    KdN = np.dot(np.dot(np.linalg.inv(np.dot(L.T, L)), L.T), I)
    KdN_norm = np.linalg.norm(KdN, axis=0).reshape(image_row*image_col, 1)
    
    NM = KdN.T/KdN_norm
    return NM

def calculate_depth(N, mask, scale = 50, threshold = 3):
    M = []
    V = []
    check_use = np.zeros((image_row * image_col))
    N = N.reshape(image_row, image_col, 3)

    for i in range(image_row):
        for j in range(image_col-1):
            if mask[i][j] == 0:
                continue
            
            tmp = np.zeros((image_row * image_col))
            tmp[i * image_col + j] = -1
            tmp[i * image_col + j + 1] = 1
            check_use[i * image_col + j] = 1
            check_use[i * image_col + j + 1] = 1

            M.append(tmp)
            # V.append(-N[i][j][0] / N[i][j][2])
            maxi = max(-N[i][j][0] / N[i][j][2], -threshold)
            mini = min(maxi, threshold)
            V.append(mini)
        
            if mask[i][j + 1] == 0:
                tmp = np.zeros((image_row * image_col))
                tmp[i * image_col + j + 1] = 1
                check_use[i * image_col + j + 1] = 1
                M.append(tmp)
                V.append(0)
                
    for i in range(image_col):
        for j in range(image_row-1):
            if mask[j][i] == 0:
                continue
            
            tmp = np.zeros((image_row * image_col))           
            tmp[j * image_col + i] = -1
            tmp[(j + 1) * image_col + i] = 1
            check_use[j * image_col + i] = 1
            check_use[(j + 1) * image_col + i] = 1
            
            M.append(tmp)
            # V.append(N[j][i][1] / N[j][i][2])
            maxi = max(N[j][i][1] / N[j][i][2], -threshold)
            mini = min(maxi, threshold)
            V.append(mini)

            if mask[j + 1][i] == 0:
                tmp = np.zeros((image_row * image_col))
                tmp[(j+1) * image_col + i] = 1
                check_use[(j+1) * image_col + i] = 1
                M.append(tmp)
                V.append(0)      

    M = np.array(M, dtype=np.float32)
    V = np.array(V, dtype=np.float32).reshape(-1, 1)
    
    # print(M.shape)
    # print(check_use.shape)
    M_filtered = M[:, check_use.astype(bool)]
    z = np.dot(np.dot(np.linalg.inv(np.dot(M_filtered.T, M_filtered)), M_filtered.T), V)

    idx = 0
    depth_map = []
    z_max = np.max(z)
    z_min = np.min(z)
    z_mid = (z_max+z_min)/2
    
    for i in range(image_row*image_col):
        if check_use[i] == 1:
            depth_map.append((z[idx][0] - z_mid)*scale / (z_max - z_min))
            idx += 1
        else:
            depth_map.append(0.0)

    depth_map = np.array(depth_map, dtype=np.float32)

    return depth_map

def get_normal_mask(normal_map):
    normal_map = normal_map.reshape(image_row, image_col, 3)
    mask = np.where(np.isnan(normal_map[:, :, 0]), 0, 1)
    
    return mask

def pixel_mask(img, thres_scale = 20):
    '''
    Mask image where pixels below the threshold
    '''
    pixel_sum = np.zeros(img[0].shape)
    for _img in img:
        pixel_sum += _img
    
    threshold = len(img) * thres_scale 
    mask = np.where(pixel_sum < threshold, 0, 1) 
    
    masked_images = [_img * mask for _img in img]
    masked_images = np.array(masked_images, dtype = np.float32)
    
    return masked_images

if __name__ == '__main__':
    cases = ["bunny", "star", "venus", "noisy_venus"]

    for case in cases:
        print("Processing " + case)
        # ============== read file & simple process noise ====================
        images = []
        LightSources = []
        for fn in sorted(os.listdir(os.path.join('./test', case))):
            f_path = os.path.join('test', case, fn)

            # read bmp
            if fn.endswith('.bmp'):
                img = read_bmp(f_path)

                # simple preprocessing img
                if case == 'noisy_venus': # denoise for noisy venus
                    img = gaussian_filter(img)
                img = low_pass_filter(img)

                images.append(img)

            # read txt
            elif fn.endswith('.txt'):
                with open(f_path) as f:
                    lines = f.readlines()
                    for line in lines:
                        # get light source from fixed pattern
                        pattern = r'\((-?\d+),(-?\d+),(-?\d+)\)'
                        ls_p = list(re.findall(pattern, line)[0])
                        sum = (float(ls_p[0])**2 + float(ls_p[1])**2 + float(ls_p[2])**2) ** 0.5
                        
                        ls_p = [float(ls_p[0]) / sum, float(ls_p[1]) / sum, float(ls_p[2]) / sum]
                        LightSources.append(ls_p)

        images = np.array(images, dtype = np.float64)
        LightSources = np.array(LightSources, dtype = np.float64)
        # ===================================================================

        # result_file = 'result'
        # result_path = os.path.join(result_file, case)
        # os.makedirs(result_path, exist_ok=True)
        # ================ calculate normal map =============================
        if case == 'noisy_venus':
            images = pixel_mask(images, thres_scale=20)
        if case == 'venus':
            images = pixel_mask(images, thres_scale=10)
        
        images = images.reshape(-1, image_row * image_col)
        normal_map = calculate_normal(images, LightSources)

        # normal_visualization(normal_map)
        # plt.savefig(result_path + '/normal_map.png')

        # =============== calculate depth map ===============================
        mask = get_normal_mask(normal_map)
        depth_map = calculate_depth(normal_map, mask, scale=30, threshold=5)
            
        # depth_visualization(depth_map)
        # plt.savefig(result_path + '/depth_map.png')

        ply_file = '/' + case + '.ply'
        save_ply(depth_map, '.' + ply_file)
        # save_ply(depth_map, result_path + ply_file)
        # show_ply(result_path + ply_file)
        
        # plt.show()
    