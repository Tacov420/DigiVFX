import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import log
def gen_gray(img):
    blank_image = img[:,:,1]
    return blank_image
def translate(image, x, y): # ref https://www.jianshu.com/p/b5c29aeaedc7
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted
def gen_mask(img, thrd = 20):
    mid = np.median(img)
    mask = np.ones(img.shape)
    # return cv2.bitwise_not(cv2.inRange(img, mid * 0.8, mid *1.2))
    mask[np.where(np.abs(img - mid) <= thrd)] = 0 # 只考慮和mid差距夠大的
    return mask
def gen_bit(img):
    mid = np.median(img)
    tmp_img = cv2.threshold(img, mid, 255, cv2.THRESH_BINARY)[1]
    return tmp_img

def calculate_error(a,b):
    mask = gen_mask(a)
    error = cv2.bitwise_xor(a,b)#.sum()
    error = np.sum(error * mask)
    return error
def find_neighbor(img_a,img_b,x,y): # 原本偏移x y 找哪個鄰居比較好
    a = gen_bit(img_a)
    b = gen_bit(img_b)
    tmp_b = translate(b,x,y)
    error = calculate_error(a,tmp_b)
    opt=(0,0)
    for (dx, dy) in [(1,1),(1,0),(1,-1),(-1,1),(-1,0),(-1,-1),(0,1),(0,-1)]:
        tmp_b = translate(b,x+dx,y+dy)
        tmp_error = calculate_error(a,tmp_b)
        if error > tmp_error:
            opt = (dx,dy)
    return opt

def align(img_a,img_b,d=0):
    if d == 0:
        return find_neighbor(img_a,img_b,0,0)
    #縮小
    small_a = cv2.resize(img_a,(img_a.shape[0]//2,img_a.shape[1]//2))
    small_b = cv2.resize(img_b,(img_b.shape[0]//2,img_b.shape[1]//2))

    x,y = align(small_a,small_b,d-1)
    dx, dy = find_neighbor(img_a,img_b,x*2,y*2)
    return (x*2 + dx, y*2 + dy)

def gen_w():
    w = np.zeros(256)
    for z in range(256):
        if z <= 127:
            w[z] = z
        else:
            w[z] = 255 - z
    return w

def picked_pixels(imgs, length, width, p):
    n = 0
    picked = []
    for i in range(0, length, length // 20):
        for j in range(0, width, width // 20):
            picked.append((i, j))
            n += 1

    R = np.zeros((n, p), dtype=int)
    G = np.zeros((n, p), dtype=int)
    B = np.zeros((n, p), dtype=int)
    for i in range(n):
        for j in range(p):
            R[i][j] = imgs[j][picked[i][0]][picked[i][1]][0]
            G[i][j] = imgs[j][picked[i][0]][picked[i][1]][1]
            B[i][j] = imgs[j][picked[i][0]][picked[i][1]][2]
    return R, G, B

# B = ln(t), n = num of pixels, p = num of photos
def debevec(Z, B, Lambda, w, n, p):
    A = np.zeros((n * p + 1 + 254, 256 + n))
    b = np.zeros(A.shape[0])

    # fill val in A
    for i in range(n):
        for j in range(p):
            A[i * p + j][Z[i][j]] = w[Z[i][j]]
            A[i * p + j][256 + i] = -w[Z[i][j]]
    
    A[n * p][127] = 1

    for i in range(254):
        A[(n * p + 1) + i][i] = 1 * w[i + 1] * Lambda
        A[(n * p + 1) + i][i + 1] = -2 * w[i + 1] * Lambda
        A[(n * p + 1) + i][i + 2] = 1 * w[i + 1] * Lambda

    # fill val in b
    for i in range(n):
        for j in range(p):
            b[i * p + j] = B[j] * w[Z[i][j]]

    x = np.linalg.lstsq(A, b, rcond = None)[0]
    g = x[:256]

    return g

def get_radiance(imgs, R, G, B, lnts, l, w, n, p):
    g_R = debevec(R, lnts, l, w, n, p)
    g_G = debevec(G, lnts, l, w, n, p)
    g_B = debevec(B, lnts, l, w, n, p)

    temp = range(256)
    plt.plot(g_R, temp, 'r')
    plt.plot(g_G, temp, 'g')
    plt.plot(g_B, temp, 'b')
    plt.show()

    radiance = np.zeros((length, width, 3))
    print("Constructing HDR...")
    hdr_progress = tqdm(total=length)
    for i in range(length):
        for j in range(width):
            se_r, sw_r, se_g, sw_g, se_b, sw_b = 0, 0, 0, 0, 0, 0
            for k in range(p):
                se_r += w[imgs[k][i][j][0]] * (g_R[imgs[k][i][j][0]] - lnts[k])
                sw_r += w[imgs[k][i][j][0]]
                se_g += w[imgs[k][i][j][1]] * (g_G[imgs[k][i][j][1]] - lnts[k])
                sw_g += w[imgs[k][i][j][1]]
                se_b += w[imgs[k][i][j][2]] * (g_B[imgs[k][i][j][2]] - lnts[k])
                sw_b += w[imgs[k][i][j][2]]
            radiance[i][j][0] = se_r / sw_r if sw_r > 0 else 0
            radiance[i][j][1] = se_g / sw_g if sw_g > 0 else 0
            radiance[i][j][2] = se_b / sw_b if sw_b > 0 else 0
        hdr_progress.update(1)
    radiance = np.exp(radiance)
    return radiance

def Reinhard(radiance, length, width, gamma, key):
    print("Start tone mapping with Reinhard...")
    Lw = radiance[:, :, 0] * 0.2126 + radiance[:, :, 1] * 0.7152 + radiance[:, :, 2] * 0.0722
    Lw_avg = np.exp((np.sum(np.log(Lw + gamma))) / (length * width))
    #print("Lw_avg = %f" % Lw_avg)
    Lm = key * Lw / Lw_avg
    Ld = np.zeros((length, width))
    for i in range(length):
        for j in range(width):
            Ld[i][j] = Lm[i][j] / (1 + Lm[i][j])
    #print(Ld)

    result = np.zeros((length, width, 3))
    tmp_progress = tqdm(total=length)
    for i in range(length):
        for j in range(width):
            for k in range(3):
                result[i][j][k] = round((Ld[i][j] * radiance[i][j][k] / Lw[i][j]) * 255)
        tmp_progress.update(1)
    cv2.imwrite("result_reinhard.jpg", cv2.cvtColor(result.astype(np.float32), cv2.COLOR_RGB2BGR))

def transfer(x, gamma = 2.2):
    if (x <= 0.05): 
        return x * 2.64
    return 1.099*pow(x, 0.9 / gamma) - 0.099;


def AdaptiveLog(radiance, length, width, base = 0.85):
    CIE = np.zeros((3,))
    result = np.zeros((length, width, 3))

    tmp_progress = tqdm(total=length)
    # Find lw_max
    Lw = radiance[:, :, 0] * 0.2126 + radiance[:, :, 1] * 0.7152 + radiance[:, :, 2] * 0.0722
    Lwmax = Lw.max()
    Ldmax = 1#Ld.max()
    #Ref: https://www.oceanopticsbook.info/view/photometry-and-visibility/from-xyz-to-rgb
    RGB2CIE = np.array([
        [0.4124564,0.3575761,0.1804375],
        [0.2126729,0.7151522,0.072175],
        [0.0193339,0.119192,0.9503041]
        ])
    CIE2RGB = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.969266, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
        ])
    for i in range(length):
        for j in range(width):
            # RGB => CIE XYZ
            CIE = np.matmul(RGB2CIE,radiance[i][j])
            ld = Ldmax / log(Lwmax + 1, 10)
            ld *= log(Lw[i][j] + 1) / log(2 + 8.0*pow(Lw[i][j] / Lwmax, log(base,0.5))) 
            xx = CIE[0] / sum(CIE)
            yy = CIE[1] / sum(CIE)
            CIE = np.array([ld / yy*xx, ld,ld / yy*(1 - xx - yy)])
            
            # CIE => RGB
            CIE = np.matmul(CIE2RGB,CIE)
            result[i][j] = np.array([int(transfer(min(max(k,0),1))*255) for k in CIE])
            
        tmp_progress.update(1)
    cv2.imwrite(f"result_log_{base}.jpg", cv2.cvtColor(result.astype(np.float32), cv2.COLOR_RGB2BGR))


# init
# img_name = ['../data/IMG_25'+str(i)+'.JPG' for i in range(48, 58)] 
img_name = [f'code/img{i:02}.jpg' for i in range(1, 14)] 
# print(img_name)

# img_name = input("Enter all the name of images(split by space):").split(" ")

imgs = [cv2.imread(i) for i in img_name]
gray_imgs = [gen_gray(i) for i in imgs] 
offsets = [(0,0)] + [align(gray_imgs[0],i,d=5) for i in gray_imgs[1:]]
align_img = [translate(i,x,y) for i, (x,y) in zip(imgs,offsets)]
print(offsets)
imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in align_img]
#lnts = [np.log(1) - np.log(x) for x in [3247, 180, 25]]    #car
lnts = np.log((np.array([13, 10, 4, 3.2, 1, 0.8, 0.3, 1/4, 1/60, 1/80, 1/320, 1/400, 1/1000]))) #desk
# lnts = np.log(np.array([1/4016, 1/2028, 1/1009, 1/501, 1/250, 1/125, 1/60, 1/30, 1/15, 1/6])) #wojak

length = imgs[0].shape[0]
width = imgs[0].shape[1]
w = gen_w()
l = 40
p = len(imgs)
R, G, B = picked_pixels(imgs, length, width, p)
n = len(R)

# start processing
radiance = get_radiance(imgs, R, G, B, lnts, l, w, n, p)    # get radiance per pixel
Reinhard(radiance, length, width, gamma = 0.2, key = 1) # tone mapping with Reinhard
AdaptiveLog(radiance, length, width, base = 0.5) # tone mapping with Adaptive Logarithmic Mapping
# Ref: https://resources.mpi-inf.mpg.de/tmo/logmap/logmap.pdf





