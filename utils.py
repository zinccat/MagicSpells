import numpy as np
import ffmpeg
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from skimage.measure import label, regionprops
from numpy.linalg import inv

def preprocess(frame):
    # frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    frame1 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # 灰度化
    frame1 = cv2.GaussianBlur(frame1, (11, 11), 20) # 高斯模糊去噪
    max_thres = 0
    max_light = max(np.max(frame1), max_thres)
    frame1 = cv2.threshold(frame1, max_light - 5, max_light, cv2.THRESH_BINARY)[1] # 设置阈值只取最亮的部分
    # frame1 = cv2.erode(frame1, None, iterations=2) # 腐蚀
    # frame1 = cv2.dilate(frame1, None, iterations=4) # 膨胀，这两步可以消除一部分小亮点
    return frame1

def mask_count(frame, thresh=100):
    labels = label(frame, background=0)
    mask_list = []
    max_count = 0
    for label_ in np.unique(labels):
        if label_ == 0: # 背景
            continue
        label_mask = np.zeros_like(frame, dtype="uint8")
        label_mask[labels == label_] = 255
        count = cv2.countNonZero(label_mask)
        if count > thresh:
            mask_list.append((label_mask, count))
            if count > max_count:
                max_count = count
    return mask_list, max_count

def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def round_center(mask):
    props = regionprops(mask)[0]
    return 4 * np.pi * props.area / props.perimeter ** 2, props.centroid[1], props.centroid[0] # 计算圆度和质心

def score(masks, max_count, recent_c, diag_l, d_thres=0.80):
    best_c = recent_c
    best_score = 0
    for mask, count in masks:
        r_score, cx, cy = round_center(mask) # 圆度评分
        if recent_c != None:
            d_score = 1 - dist(cx, cy, *recent_c) / diag_l # 与上一帧圆心的距离评分
        else:
            d_score = 1
        c_score = count / max_count # 亮点像素数量评分
        if r_score + d_score + c_score > best_score:
            best_c = (cx, cy)
    return best_c
    
def ana_traj(video, list_time,d_thres=0.80):
    '''
    返回 video 的轨迹列表
    '''
    center_list = []
    time_list = []
    recent_c = None
    diag_l = np.sqrt(video.shape[1] ** 2 + video.shape[2] ** 2)
    for i in range(video.shape[0]):
        # frame = preprocess(video[i])
        frame = video[i]
        mask_list, max_count = mask_count(frame)
        recent_c = score(mask_list, max_count, recent_c, diag_l, d_thres)
        if recent_c != None:
            center_list.append(recent_c)
            time_list.append(list_time[i])
        # center_list.append(recent_c)
    return center_list, time_list

def bayes_filter(traj, list_time, var=0.1):
    traj_ori = np.array(traj) 
    traj_kf = traj_ori.copy()
    for i in range(len(traj)):
        if i < 2:
            traj_kf[i] = traj_ori[i]
            continue
        est_1 = traj_ori[i]
        est_2 = traj_kf[i-1] + (traj_kf[i-1] - traj_kf[i-2]) * list_time[i] / list_time[i-1]
        var_1 = var
        var_2 = 2 * var * list_time[i]**2 / list_time[i-1]**2
        # merge two estimates using MLE
        traj_kf[i] = (est_1 / var_1 + est_2 / var_2) / (1/var_1 + 1/var_2)
    return traj_kf

def resize(img, padding=True):
    row = ~(img==255).all(1)
    col = ~(img==255).all(0)
    img1 = img[row]
    img1 = img1[:,col]
    if padding:
        y, x = img1.shape
        if y < x:
            j = int(0.35 * y) # 这个值必须大于 0.25
            i = int(3 * x * (y + 2 * j) / 4 / y - x / 2)
        else:
            i = int(0.35 * x)
            j = int(y * (x + 2 * i) / 3 / x - y / 2)
        img1 = np.concatenate(
            (255 * np.ones((y, i)), img1, 255 * np.ones((y, i))),
            axis=1
        )
        img1 = np.concatenate(
            (255 * np.ones((j, x+2*i)), img1, 255 * np.ones((j, x+2*i))),
            axis=0
        ).astype(np.uint8)
    return img1

if __name__ == "__main__":
    img = cv2.imread("1654535263.5655363_filtered.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = resize(img)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    plt.figure()
    plt.imshow(img)
    plt.show()
    