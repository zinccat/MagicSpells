import numpy as np
from matplotlib import pyplot as plt
import cv2
from pygame import mixer

from utils import ana_traj, bayes_filter, resize

import numpy as np
from time import time, sleep

predict = True
if predict:
    import torch
    from model import load_model
    model = load_model(path='mnist.pkl')
    print('model loaded!')

mixer.init()

while True:
  camera = cv2.VideoCapture(2)  # 参数0表示第一个摄像头
  # 判断视频是否打开
  if (camera.isOpened()):
      print('摄像头成功打开1')
  else:
      print('摄像头未打开')
  # 测试用,查看视频size
  size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
  print('size:'+repr(size))

  es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
  background = None

  list_traj = []
  list_time = []
  t_old = t_start = time()
  idx = 0
  limit = 10000
  started = False
  flag = False
  while True:  # len(list_traj) < 200:
      # t_old = time()
      # 读取视频流
      grabbed, frame_lwpCV = camera.read()
      # 对帧进行预处理，先转灰度图，再进行高斯滤波。
      # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
      gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
      gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)

      # 将前2s内容设置为整个输入的背景
      # if background is None:
      if time() - t_start < 2:
          if background is None:
              background = gray_lwpCV
      # 对于每个从背景之后读取的帧都会计算其与北京之间的差异，并得到一个差分图（different map）。
      # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
      diff = gray_lwpCV - background
      frame_avg = np.average(frame_lwpCV, axis=2).astype(np.uint8)
      thresh = cv2.threshold(frame_avg, np.max(
          frame_avg)-5, 255, cv2.THRESH_BINARY)[1]
      diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]  # 二值化阈值处理
      diff = diff * np.sign(thresh)
      diff = cv2.erode(diff, None, iterations=2)
      diff = cv2.dilate(diff, es, iterations=2)  # 形态学膨胀

      # 显示矩形框
      # contours, hierarchy = cv2.findContours(
          # diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 该函数计算一幅图像中目标的轮廓
      # for c in contours:
      #     # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
      #     if cv2.contourArea(c) < 1500:
      #         continue
      #     (x, y, w, h) = cv2.boundingRect(c)  # 该函数计算矩形的边界框
      #     cv2.rectangle(frame_lwpCV, (x, y), (x+w, y+h), (0, 255, 0), 2)
      diff_sum = np.sum(diff)
      cv2.imshow('dis', diff)
      t = time()
      if not flag and t - t_start > 2:
          print("\nWarm up ended!")
          flag = True
      print("\r logged {} time since last log {} s".format(
          len(list_traj), t - t_old), end='')
      if not started and t - t_start > 2 and diff_sum > 1.1*limit:
          print("\nStarted logging!")
          started = True
      if diff_sum > 1.1*limit and started:  # 仅添加有内容的帧到轨迹列表, 可以考虑调整阈值
          list_traj.append(diff)
          list_time.append(t-t_old)
          t_old = t
      elif started:
          pass
      else:
          limit = max(diff_sum, limit)
          continue
      if started and t - t_old > 2:
          break
  # When everything done, release the capture
  camera.release()
  cv2.destroyAllWindows()

  # remove first k frames
  k = 3
  for i in range(k):
      list_traj.pop(0)
      list_time.pop(0)
  # remove last k frames
  for i in range(k):
      list_traj.pop()
      list_time.pop()

  list_traj = np.array(list_traj)
  center_list, time_list = ana_traj(list_traj, list_time, d_thres=0.1)

  x = [i[0] for i in center_list]
  y = [i[1] for i in center_list]

  # set figure size
  plt.figure(figsize=(9, 12))
  plt.plot(x, y, 'o-', linewidth=20)

  height = 720
  width = 1280
  # set axis limits
  plt.xlim(0, width)
  plt.ylim(0, height)
  plt.gca().invert_xaxis()
  plt.gca().invert_yaxis()
  # unset axis
  plt.axis('off')
  plt.savefig('{}_ori.png'.format(t), dpi=50)

  traj_filtered = bayes_filter(center_list, time_list)

  x = [i[0] for i in traj_filtered]
  y = [i[1] for i in traj_filtered]

  # set figure size
  plt.figure(figsize=(9, 12))
  plt.plot(x, y, 'o-', linewidth=20)

  height = 720
  width = 1280
  # set axis limits
  plt.xlim(0, width)
  plt.ylim(0, height)
  plt.gca().invert_xaxis()
  plt.gca().invert_yaxis()
  plt.axis('off')
  plt.savefig('{}_filtered.png'.format(t), dpi=50)

  if predict:
      img = cv2.imread('{}_filtered.png'.format(t))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      img = resize(img, True)
      img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
      
      img[img == 255] = 0
      img[img != 0] = 255
      img = img/255
      img = (img-0.5)/0.5
      img = img.reshape(1, 1, 28, 28)
      img = torch.tensor(img)
      img = img.float()
      number = model(img).argmax().item()
      print("\nPredicted: ", number)
      if number == 8:
        mixer.music.load(r"E:\CloudMusic\8.mp3")
        mixer.music.play()
        sleep(60)
        mixer.music.stop()
      elif number == 2:
        mixer.music.load(r"E:\CloudMusic\2.mp3")
        mixer.music.play()
        sleep(5)
        mixer.music.stop()
      elif number == 3:
        mixer.music.load(r"E:\CloudMusic\3.mp3")
        mixer.music.play()
        sleep(5)
        mixer.music.stop()
      elif number == 5:
        mixer.music.load(r"E:\CloudMusic\5.mp3")
        mixer.music.play()
        sleep(5)
        mixer.music.stop()
      elif number == 6:
        mixer.music.load(r"E:\CloudMusic\6.mp3")
        mixer.music.play()
        sleep(5)
        mixer.music.stop()
