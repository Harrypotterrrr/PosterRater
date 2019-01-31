#!/usr/bin/env python
# coding: utf-8

# In[12]:


"""Data Aumentation
   1. 翻转变换 flip
   2. 随机修剪 random crop
   3. 色彩抖动 color jittering
   4. 平移变换 shift
   5. 尺度变换 scale
   6. 对比度变换 contrast
   7. 噪声扰动 noise
   8. 旋转变换/反射变换 Rotation/reflection
"""

from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import threading, os, time
from multiprocessing import Pool

# import logging
# logger = logging.getLogger(__name__)
# ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageProcess:
    """
    Eight ways to augment image data
    """


    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")
    
    @staticmethod
    def saveImage(image, path):
        image.save(path)
        
    @staticmethod
    def checkDir(src_path, dst_path):
        if not os.path.isdir(src_path):
            print(src_path, "is not directory!")
            return False
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            print("make new", dst_path, "path!")   
        return True
        
    @staticmethod
    def imageResize(image, width=200, height=280, mode=Image.BICUBIC):
        """
        resize the image to specific size
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL class image
        :return: image after resize
        """
        return image.resize((width, height), mode)

    @staticmethod
    def randomRotation(image, mode=Image.BICUBIC):
        """
        rotate image at random angle
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL class image
        :return: image after rotation
        """
        random_angle = np.random.randint(0, 360)
        return image.rotate(random_angle, mode)

    @staticmethod
    def randomCrop(image):
        """
        considering the size of pic (200, 280), use windows ranged from 40 to 70 to crop randomly
        :param image: PIL class image
        :return: image after crop

        """
        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_size = np.random.randint(160, 200)
        random_region = (
            (image_width - crop_win_size) >> 1, 
            (image_height - crop_win_size) >> 1, 
            (image_width + crop_win_size) >> 1,
            (image_height + crop_win_size) >> 1
        )
        image = image.crop(random_region)
        return ImageProcess.imageResize(image)


    @staticmethod
    def randomColor(image):
        """
        use color jittering
        :param image: PIL class image
        :return: image of
        """
        # No restriction to random_factor, 1.0 is origin
        
        # adjust the saturation
        random_factor = np.random.uniform(0.7, 1.6)
        minus = np.random.choice([1, -1])
        random_factor *= minus
        color_image = ImageEnhance.Color(image).enhance(random_factor)  
        
        # adjust the brightness
        random_factor = np.random.uniform(0.6, 1.3) 
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  
        
        # adjust the contrastion
        random_factor = np.random.uniform(0.4, 1.8)
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
        
        # adjust the sharpness
        random_factor = np.random.uniform(0, 2)
        sharpness_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
        
        return sharpness_image

    @staticmethod
    def randomGaussian(image, mean=0.0, stddev=0.3):
        """
        use Gaussian nosiy to process image
        :param image:
        :return:
        """

        def gaussianNoisy(im, mean=0.0, stddev=0.3):
            """
            sub function
            :param im: image in 1 dimension
            :param mean: mean
            :param stddev: stddev
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, stddev)
            return im

#       # convert PIL image to array (np.asarray will request no memory compared to np.array)
#         img = np.asarray(image)
#       # set writable flag to True
#         img.flags.writeable = True

        img = np.array(image)

        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, stddev)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, stddev)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, stddev)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img))

    @staticmethod
    def imgProcess(file, src_path, dst_path, times=20):
        """
        image process for each file
        """
        opsList = {"randomRotation", "randomCrop", "randomColor", "randomGaussian"}
        funcMap = {"randomRotation": ImageProcess.randomRotation,
               "randomCrop": ImageProcess.randomCrop,
               "randomColor": ImageProcess.randomColor,
               "randomGaussian": ImageProcess.randomGaussian
               }

        file_path = src_path + file
        if not os.path.isfile(file_path):
            print("image doesn't exist!")
            return
        im = ImageProcess.openImage(file_path)
        im_name = file.split('.')[0]

        for i in range(times):
            im_cur = im
            for func in funcMap.values():
                im_cur = func(im_cur)
            ImageProcess.saveImage(im_cur, dst_path+im_name+"_"+str(i)+".png")
        print(file+" process finish!")
    
    @staticmethod
    def multiProcess(src_path, dst_path, times=20):
        """
        The solution of multi-processs
        """
        if ImageProcess.checkDir(src_path, dst_path) is False:
            return
        
        files = os.listdir(src_path)
        print("file's count:", len(files))
        pool = Pool()
        for file in files:
            pool.apply_async(ImageProcess.imgProcess, args=(file, src_path, dst_path, times))
        pool.close()
        pool.join()

    @staticmethod
    def multiThread(src_path, dst_path, times=20):
        """
        The solution of multi-thread
        """
        if ImageProcess.checkDir(src_path, dst_path) is False:
            return

    #     root, dirs, files = os.walk(src_path)
        files = os.listdir(src_path)

        print(len(files))
        for file in files:
            t = threading.Thread(target=ImageProcess.imgProcess, args=(file, src_path, dst_path, times))
    #         t.setDaemon(True)
            print(file + " is processing...")
            t.start()
            time.sleep(0.2)
        t.join()

    @staticmethod
    def singleThread(src_path, dst_path, times=20):
        """
        The solution of single-process/thread
        """
        if ImageProcess.checkDir(src_path, dst_path) is False:
            return

    #     root, dirs, files = os.walk(src_path)
        files = os.listdir(src_path)

        print(len(files))
        for file in files:
            print(file + " is processing...")
            ImageProcess.imgProcess(file, src_path, dst_path, times)


# In[13]:


if __name__ == '__main__':
    
    src_path = "./data/poster_negative/"
    dst_path = "./data/tmp/"
    
    ImageProcess.multiProcess("./data/poster_positive/", "./data/tmp/", 5)
#     multiThread(r"./data/poster_positive/", r"./data/tmp/")
#     singleThread(r"./data/poster_negative/", r"./data/tmp2/")
    
    print("end")


# In[ ]:


path = './data/poster_positive/'
test_pic_list = os.listdir(path)
print(len(test_pic))
test_im = Image.open(path+test_pic[0])
print(np.array(test_im).shape)
opsList = {"randomRotation", "randomCrop", "randomColor", "randomGaussian"}
funcMap = {"randomRotation": ImageProcess.randomRotation,
       "randomCrop": ImageProcess.randomCrop,
       "randomColor": ImageProcess.randomColor,
       "randomGaussian": ImageProcess.randomGaussian
       }

im_name = test_pic[0].split('.')[0]

im_cur = test_im
for key, func in funcMap.items():
    im_cur = func(im_cur)
    print(key, np.array(im_cur).shape)

