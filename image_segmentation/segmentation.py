import cv2
import numpy as np
from PIL import Image

class SimplyProjection:
    """ 文本行检测算法
        比较原始的方法，通过水平投影的方式确定行和行的边界，通过垂直投影的方式确定文本的左右边界，要注意公式行高可能比文字的范围大。
        TODO 输入一张图片 输出切割后的文本行
    """
    def __init__(self):
        pass

    def theory_visual(self):
        """
            简单可视化投影法的原理
        """
        img=cv2.imread('../images/724008042.png')
        #灰度图片进行二值化处理
        GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #阈值筛选，130是超参数
        ret,thresh1=cv2.threshold(GrayImage,130,255,cv2.THRESH_BINARY)
        ret,thresh2=cv2.threshold(GrayImage,130,255,cv2.THRESH_BINARY)

        #水平投影
        (h,w)=thresh1.shape
        a = [0 for z in range(0, h)]
        for j in range(0,h):
            for i in range(0,w):
                if thresh1[j,i]==0:
                    a[j]+=1
                    # 可视化美观用
                    thresh1[j,i]=255
        # 可视化美观用
        for j in range(0,h):
            for i in range(0,a[j]):
                thresh1[j,i]=0

        #垂直投影
        a = [0 for z in range(0, w)]
        for j in range(0,w):
            for i in range(0,h):
                if  thresh2[i,j]==0:
                    a[j]+=1
                    # 可视化美观用
                    thresh2[i,j]=255
        for j in range(0,w):
            for i in range((h-a[j]),h):
                # 可视化美观用
                thresh2[i,j]=0

        #展示图片
        cv2.imshow("src",img)
        cv2.imshow('img',thresh1)
        cv2.imshow('img2',thresh2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process(self, image_path, background_edge=130, cut_edge=5):
        """
            水平切割的大概思路如下，以投影后图片纵向像素点数量的列表[3,4,5,0,0,0,5,6,7]为例，找到每两个连续非0序列([3,4,5]和[5,6,7]之间
            0序列的中点，以该坐标切割图片，即横向切割为[3,4,5,0]和[0,0,5,6,7]两张图片，设定0序列的长度小于5个像素点(超参)则不进行切割。
            #TODO 尽可能减少不必要的图片空白
        :param image_path: str格式，输入图片路径
        :param background_edge: int格式，背景像素点灰度深浅的二值化阈值(0-255)
        :param cut_edge: int格式，行和行之间至少需要隔多少个像素点的空白
        """
        img = cv2.imread(image_path)
        # 文件名提取
        img_name = image_path.split("/")[-1]
        img_name = img_name.split('.')[0]
        img_file_path = "/".join(image_path.split("/")[:-1])
        # 灰度图片进行二值化处理
        GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 阈值筛选
        ret, thresh1 = cv2.threshold(GrayImage, background_edge, 255, cv2.THRESH_BINARY)
        ret, thresh2 = cv2.threshold(GrayImage, background_edge, 255, cv2.THRESH_BINARY)

        # 水平切分
        (h, w) = thresh2.shape
        width = [0 for z in range(0, w)]
        for j in range(0,w):
            for i in range(0,h):
                if thresh2[i,j]==0:
                    width[j] += 1
        left_index, right_index = 0, w-1
        while width[left_index] == 0 and left_index < right_index:
            left_index += 1
        while width[right_index] == 0 and left_index < right_index:
            right_index -= 1
        # 切割图片左右的空白
        crop_img = img[:, left_index:right_index]

        # 垂直方向，图片分行切割
        height = [0 for z in range(0, h)]
        for j in range(0,h):
            for i in range(0,w):
                if thresh1[j,i]==0:
                    height[j]+=1

        total_split = []
        sequence_start, zero_start, index = 0, 0, 0
        while index < h:
            if height[index] == 0:
                zero_start = index
                # 获得空白序列
                while index < h and height[index] == 0:
                    index += 1
                # 取行和行中点进行切割
                if index - zero_start > cut_edge and index < h:
                    total_split.append([sequence_start, zero_start + (index - zero_start) // 2])
                    sequence_start = zero_start + ((index - zero_start) // 2 + 1)
            else:
                index += 1
        total_split.append([sequence_start, h-1])

        # 写切割后的图片
        for img_i, img_split in enumerate(total_split):
            sub_img = crop_img[img_split[0]:img_split[1], :]
            sub_img_name = img_file_path + "/" + img_name + "-" + str(img_i) + ".png"
            cv2.imwrite(sub_img_name, sub_img)



if __name__ == "__main__":
    segment = SimplyProjection()
    #segment.theory_visual()
    segment.process('../images/2019math.png')
