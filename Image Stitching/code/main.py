import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import cdist
import math
# LA.eig(a)


def gen_gray(img):
    # gray_image = img[:,:,1]
    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray_image
def translate(image, x, y): # ref https://www.jianshu.com/p/b5c29aeaedc7
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

def genWindow(ori,i,j,bins,mag):
    window = np.zeros((bins,))
    if(i>=ori.shape[0]): return window
    if(j>=ori.shape[1]): return window
    for dx in range(5):
        if(i+dx<0): continue
        if(i+dx>=ori.shape[0]): break

        for dy in range(5):
            if(j+dy<0): continue
            if(j+dy>=ori.shape[1]): break

            window[ori[i+dx][j+dy]-1] += mag[i+dx][j+dy]# * (1-((dx**2+dy**2)/18)**(1/2))  # TODO weighted function
    return window

def genFeatureDescriptor(img, bins=10, k=0.04): # assume window function is neighber 3*3
    gray = gen_gray(img)
    # blurredx = cv2.GaussianBlur(gray, (5, 1), 0)
    # blurredy = cv2.GaussianBlur(gray, (1, 5), 0)
    
    # sobelx = cv2.Sobel(blurredx,cv2.CV_64F,1,0,ksize=5)  # x
    # sobely = cv2.Sobel(blurredy,cv2.CV_64F,0,1,ksize=5)  # y
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sobely, sobelx = np.gradient(gray_blur)
    # print(sobelx)
    Ix2 = np.multiply(sobelx ,sobelx)
    Ixy = np.multiply(sobelx ,sobely)
    Iy2 = np.multiply(sobely ,sobely)
    
    Sx2 = cv2.GaussianBlur(Ix2, (5, 5), 0)
    Sxy = cv2.GaussianBlur(Ixy, (5, 5), 0)
    Sy2 = cv2.GaussianBlur(Iy2, (5, 5), 0)
    R = np.zeros(img.shape[:2])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            M = np.array([[Sx2[i][j],Sxy[i][j]],
                [Sxy[i][j],Sy2[i][j]]])
            R[i][j] = LA.det(M) - k*(np.trace(M))**2
    # print(R[0][0])
    # print(R.max(),R.min())
    thresh = 0.1*cv2.dilate(R,None).max()
    Features = []

    # Find local maxima
    
    Features = []
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            is_max = True
            for x in range(-1,2):
                for y in range(-1,2):
                    if(R[i][j]<R[i+x][j+y]):
                        is_max = False
                        break
                if not is_max:
                    break
            if is_max and R[i][j] > thresh:
                Features.append([i,j])
    
    # Located all features
    # for j in range(0, img.shape[0]):
    #     for i in range(0, img.shape[1]):
    #         if(R[j,i] > thresh):
    #             # image, center pt, radius, color, thickness
    #             Features.append((i, j))
    #             cv2.circle(corner_image, (i, j), 1, (0,255,0), 1)
    mag = (Ix2 + Iy2) ** (1/2)
    orient = np.arctan(sobely / (sobelx+1e-6)) * (180 / np.pi)
    orient[sobelx < 0] += 180  # 省
    orient = (orient + 360) % 360
    bin_x = np.linspace(0,360,bins+1) # 0,10,20,....350 360
    orient_bins = np.digitize(orient, bin_x)
    for ind in range(len(Features)):
        i,j = Features[ind]
        orient_weighted = np.zeros((bins,))
        descriptor = []
        for dx in range(-2,2):  # 取4x4個window的子向量 一個windows
            for dy in range(-2,2):
                descriptor.append(genWindow(orient_bins,i+4*dx,j+4*dy,bins,mag))
                orient_weighted += descriptor[-1]
                # for _ in range(len(orient_weighted)):
                #     orient_weighted[_] += descriptor[-1][_]
        descriptor = np.concatenate(descriptor) # bin*16(windows) 維
        Features[ind].append(descriptor)
        Features[ind].append((np.argmax(orient_weighted, axis=0) * (360//bins), orient_weighted.max()))
        
    corner_image = np.copy(img)
    left_feature = []
    right_feature = []

    for i,j,_,(t,x) in Features:
        # print(i,j,t,x)
        # print(np.pi,np.cos(t/np.pi),np.cos(t))
        # print(np.cos(t/360 * np.pi))
        if(j<img.shape[1]//2):
            left_feature.append((i,j,_,(t,x)))
        else:
            right_feature.append((i,j,_,(t,x)))

        cv2.circle(corner_image, (j, i), 1, (0,255,0), int(x)//5000+1)
        cv2.arrowedLine(corner_image, (j, i),(j+int(10*np.cos(t/360*np.pi)), i+int(10*np.sin(t/360*np.pi))),(0,0,255))
        # (j+x*math.cos(t/np.pi), i+x*math.sin(t/np.pi))

    
    return corner_image, left_feature, right_feature

# Ref https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/
# def genFeatureDescriptor(img,Ix,Iy):

    # gray_img = gen_gray(img)
    # scaled_imgs = [cv2.resize(gray_img,(img.shape[0]//(2**i),img.shape[1]//(2**i))) for i in range(4)]
    # pool = [[i] for i in scaled_imgs]
    # for i in pool:
    #     for _ in range(4):
    #         i.append(cv2.GaussianBlur(i[-1], (5, 5), 0))
    # scaled_DoG = [] # Unblur -blur
    # for i in pool:
    #     tmp = []
    #     for j in range(4):
    #         tmp.append(i[j]-i[j+1])
    #     scaled_DoG.append(tmp)
    # for i in scaled_DoG:
    #     for j in i:
    #         cv2.imshow('1',j)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

    # Determine sample points.
    # Determine the gradient magnitude and orientation of each sample point.
    # Create an orientation histogram.
    # Extract dominant directions.

# def Euv()
# # init

def matchFeature(l_fea, r_fea, thrd=0.8): # Brute-Force sort finding
    # Ref https://www.geeksforgeeks.org/python-distance-between-collections-of-inputs/
    l_vec = [i[2] for i in l_fea]
    r_vec = [i[2] for i in r_fea]
    l_keypoints = [i[:2] for i in l_fea]
    r_keypoints = [i[:2] for i in r_fea]
    # print(l_vec[:5])
    # print(len(l_vec),len(r_vec))
    dist_table = cdist(l_vec, r_vec)
    # print(dist_table)
    sorted_dist = np.argsort(dist_table, axis=1)
    matches = []

    for x, ind in enumerate(sorted_dist): # 幫第x號點找match
        dist_first = dist_table[x][ind[0]]
        dist_second = dist_table[x][ind[1]]
        if(dist_first/dist_second < thrd):
            matches.append((x,ind[0]))

    # print(l_keypoints, r_keypoints, matches)
    return np.array(l_keypoints), np.array(r_keypoints), matches


def cylindrical_coor(x, y, f, w, h):
    return np.column_stack((f * np.arctan((x - w/2)/f) + f * np.arctan((w/2)/f), f * ((y - h/2) / np.sqrt(np.square(x - w/2) + f**2)) + h/2))

def compute_alignments(l_img_pts, r_img_pts, matches, f, w, h, k = 8000, thrd = 5):
    l_pts = cylindrical_coor(l_img_pts[:, 1], l_img_pts[:, 0], f, w, h)
    r_pts = cylindrical_coor(r_img_pts[:, 1], r_img_pts[:, 0], f, w, h)

    max_inliner = 0
    best_shift = None
    N = len(matches)

    for r in range(k):
        index = random.randint(0, len(matches) - 1)
        theta = l_pts[matches[index][1]] - r_pts[matches[index][0]]

        inliner = 0
        for i in range(N):
            if i != index:
                dist = np.sqrt(np.sum(np.square(l_pts[matches[i][1]] - (r_pts[matches[i][0]] + theta))))
                if dist < thrd:
                    inliner += 1

        if inliner > max_inliner:
            max_inliner = inliner
            best_shift = theta
    
    return best_shift

def cylindrical_warping(img, f):
    w = len(img[0])
    h = len(img)
    cyl_img = np.zeros((h, math.ceil(f * np.arctan((w/2) / f)) * 2, 3), dtype=int)
    for i in range(h):
        for j in range(w):
            x, y = cylindrical_coor(j, i, f, w, h)[0]
            cyl_img[round(y)][round(x)] = img[i][j]
    return cyl_img

def align_no_blending(imgs, shifts):
    print("No Blending")
    w = len(imgs[0][0])
    h = len(imgs[0])    
    res_img = np.zeros((h, w + math.ceil(np.sum(shifts[:, 0])), 3), dtype=int)

    offset = np.array([0, 0])
    for k in range(len(imgs) - 1, -1, -1):
        for i in range(h):
            for j in range(w):
                if i + offset[1] >= 0 and i + offset[1] < h:
                    res_img[int(i + offset[1])][int(j + offset[0])] = imgs[k][i][j]
        if(k > 0): offset = offset + shifts[k - 1]
    
    res_img = res_img.astype(np.uint8)
    cv2.imshow('Result', res_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("pano_no.jpg", res_img)
    return

def align_linear_blending(imgs, shifts):
    print("Linear Blending")
    w = len(imgs[0][0])
    h = len(imgs[0])    
    res_img = np.zeros((h, w + math.ceil(np.sum(shifts[:, 0])), 3), dtype=int)

    offset = np.array([0, 0])
    border = 0
    for k in range(len(imgs) - 1, -1, -1):
        for i in range(h):
            for j in range(w):
                x = int(j + offset[0])
                y = int(i + offset[1])
                if y >= 0 and y < h:
                    if x >= border:
                        res_img[y][x] = imgs[k][i][j]
                    else:
                        res_img[y][x] = (imgs[k][i][j] * (j / (border - offset[0]))) + (res_img[y][x] * (1 - (j / (border - offset[0])))).astype(int)
        
        border = offset[0] + w
        if(k > 0): offset = offset + shifts[k - 1]
    
    res_img = res_img.astype(np.uint8)
    cv2.imshow('Result', res_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("pano_linear.jpg", res_img)
    return

def align_poisson_blending(imgs, shifts):
    print("Poisson Blending")
    w = len(imgs[0][0])
    h = len(imgs[0])    
    res_img = np.zeros((h, w + math.ceil(np.sum(shifts[:, 0])), 3), dtype=int)
    d = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # iterative optimization
    min_diff = np.zeros((3))
    for round in range(10):
        offset = np.array([0, 0])
        border = 0
        for k in range(len(imgs) - 1, -1, -1):
            for i in range(h):
                for j in range(w):
                    x = int(j + offset[0])
                    y = int(i + offset[1])

                    if y >= 0 and y < h:
                        if not np.any(res_img[y][x]) or x >= border:    # no value yet or outside the border
                            res_img[y][x] = imgs[k][i][j]
                        else:
                            prev_neighbors = np.zeros((3))
                            cur_neighbors = np.zeros((3))
                            for u in range(4):    
                                nx = x + d[u][1]
                                ny = y + d[u][0]
                                if ny >= 0 and ny < h and nx >= 0 and np.any(res_img[ny][nx]):
                                    prev_neighbors = prev_neighbors + res_img[ny][nx] 
                                elif np.any(res_img[y][x]):
                                    prev_neighbors = prev_neighbors + res_img[y][x]
                                else:
                                    prev_neighbors = prev_neighbors + imgs[k][i][j]

                                sx = j + d[u][1]
                                sy = i + d[u][0]
                                if sy >= 0 and sy < h and sx >= 0 and sx < w and np.any(imgs[k][sy][sx]):
                                    cur_neighbors = cur_neighbors + imgs[k][sy][sx]
                                else:
                                    cur_neighbors = cur_neighbors + imgs[k][i][j]

                            val = (((4 * imgs[k][i][j] - cur_neighbors) + prev_neighbors) / 4).astype(int)
                            diff = 4 * val - prev_neighbors - ((4 * imgs[k][i][j] - cur_neighbors) + prev_neighbors)
                            if round == 0 or np.sum(np.square(diff)) <= np.sum(np.square(min_diff)):
                                min_diff = diff
                                res_img[y][x] = val
            
            border = offset[0] + w
            if(k > 0): offset = offset + shifts[k - 1]
    
    res_img = res_img.astype(np.uint8)
    cv2.imshow('Result', res_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("pano_poisson.jpg", res_img)
    return

def align_images(imgs, shifts, blending = "linear"):    
    if blending == "no":
        align_no_blending(imgs, shifts)
    elif blending == "linear":
        align_linear_blending(imgs, shifts)
    elif blending == "poisson":
        align_poisson_blending(imgs, shifts)
    return   



img_name = [f'./parrington/prtn{i:02}.jpg' for i in range(18)]        #parrington
# img_name = [f'./5/DSC0{i}.jpg' for i in range(4108, 4119) if i != 4112]     #folder 5
# img_name = [f'./6/DSC0{i}-transformed.jpeg' for i in range(4119, 4124)]     #folder 6

# img_name = input("Enter all the name of images(split by space):").split(" ")

#ref : https://medium.com/data-breach/introduction-to-harris-corner-detector-32a88850b3f6
imgs = [cv2.imread(i) for i in img_name]

print("Start Corner Detection")
marked_imgs = [genFeatureDescriptor(i) for i in imgs]  # (回傳圖片, x/y方向gradient)
# genFeatureDescriptor(*marked_imgs[0])
print("Start Feature Matching")
all_l_kps = []
all_r_kps = []
all_matches = []
for ind in range(len(marked_imgs)-1):
    l_kps, r_kps, matches = matchFeature(marked_imgs[ind][1],marked_imgs[ind+1][2])  # 從右邊拍到左邊
    all_l_kps.append(l_kps)
    all_r_kps.append(r_kps)
    all_matches.append(matches)
    ### 畫圖用
    '''
    l_image = np.copy(imgs[ind])
    r_image = np.copy(imgs[ind+1])
    for l,r in matches:
        cv2.circle(l_image, l_kps[l][::-1], 1, (0,255,0), 5)
        cv2.circle(r_image, r_kps[r][::-1], 1, (0,255,0), 5)

    Hori = np.concatenate((r_image, l_image), axis=1)
    cv2.imshow('HORIZONTAL', Hori)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    ###
    # image matching + blending

print("Start Alignments Computing")
focal_len = [704.916, 706.286, 705.849, 706.645, 706.587, 705.645, 705.327, 704.696, 703.794, 704.325, 704.696, 703.895, 704.289, 704.676, 704.847, 704.537, 705.102, 705.576]  #parrington
# focal_len = [1624.21, 1672.4, 1668.56, 1666.59, 1663.37, 1660.57, 1657.89, 1655.08, 1652.82, 1625.8]    # folder 5
# focal_len = [849.146, 843.595, 837.83, 833.782, 855.86]     #folder 6
f = np.mean(focal_len)
print("focal len = ", f)
shifts = []
for i in range(len(all_l_kps)):
    shifts.append(compute_alignments(all_r_kps[i], all_l_kps[i], all_matches[i], f, len(imgs[0][0]), len(imgs[0])))
shifts = np.array(shifts)
print(shifts)

print("Start Image Matching")
cyl_imgs = np.array([cylindrical_warping(img, f) for img in imgs])
align_images(cyl_imgs, shifts, "poisson")