import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图片
image = cv2.imread('F:/Image/SEM/1/2.7Ti1.3B/35000-4.png', cv2.IMREAD_GRAYSCALE)
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')  # 使用灰度色图显示
plt.title("Original Grayscale Image")  # 图像标题
plt.axis('off')  # 关闭坐标轴


# 阈值处理，将图像转换为二值图像，亮的析出相变为白色，基体变为黑色
_, thresholded = cv2.threshold(image, 105, 255, cv2.THRESH_BINARY)

# 计算白色区域的像素数
white_pixels = np.sum(thresholded == 255)

# 计算整个图片的像素数
total_pixels = thresholded.size

# 计算白色区域占整个图片的比例
white_ratio = white_pixels / total_pixels

print(f"白色区域占整个图片的比例: {white_ratio:.4f}")

# 显示阈值处理后的图像
plt.subplot(1, 2, 2)
plt.imshow(thresholded, cmap='gray')
plt.title(f"Percentage of Precipitations: {white_ratio*100:.4f}%")
plt.axis('off')  # 关闭坐标轴
plt.show()



