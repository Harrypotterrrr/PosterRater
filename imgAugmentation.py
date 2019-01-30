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
    def randomRotation(image, mode=Image.BICUBIC):
        """
        rotate image at random angle
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL class image
        :return: image after rotation
        """
        random_angle = np.random.randint(1, 360)
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
        return image.crop(random_region)


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


def oneProcess(file, src_path, dst_path, times=20):
    
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
        

def threadProcess(src_path, dst_path):
    if not os.path.isdir(src_path):
        print("src_path is not directory!")
        return
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
        print("make new destination path!")
        return
    
#     root, dirs, files = os.walk(src_path)
    files = os.listdir(src_path)

    
    print(len(files))
    for file in files:
        t = threading.Thread(target=oneProcess, args=(file, src_path, dst_path, ))
        print(file + " is processing...")
        t.start()
    t.join()
        

if __name__ == '__main__':
    threadProcess(r"./data/poster_positive/", r"./data/tmp/")
    print("end")