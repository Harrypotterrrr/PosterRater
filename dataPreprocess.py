#!/usr/bin/env python
# coding: utf-8

# In[16]:


import glob
from PIL import Image

path = './data/poster/'

# get the list of png
imlist = glob.glob(path + "*.png")

im = Image.open(imlist[0])
min_width = im.size[0]
max_width = im.size[0]
min_height = im.size[1]
max_height = im.size[1]
min_hw = im.size[1] / im.size[0]
max_hw = im.size[1] / im.size[0]

# height-width ratio
hw_mean = 0

for item in imlist:
    im = Image.open(item)
    
    # find the minimum
    if im.size[0] < min_width and im.size[1] < min_height:
        min_width = im.size[0]
        min_height = im.size[1]
    
    # find the maximum
    if im.size[0] > max_width and im.size[1] > max_height:
        max_width = im.size[0]
        max_height = im.size[1]
    hw_mean += im.size[1] / im.size[0]
    if (im.size[1] / im.size[0]) < min_hw:
        min_hw = im.size[1] / im.size[0]
    if (im.size[1] / im.size[0]) > max_hw:
        max_hw = im.size[1] / im.size[0]

print("min width: " + str(min_width) + " min height: " + str(min_height))
print("max width: " + str(max_width) + " max height: " + str(max_height))
print("min h/w: " + str(min_hw) + "max h/w: " + str(max_hw))
print("mean h/w: " + str(hw_mean / len(imlist)))


# In[24]:


# 压缩图片大小
standard_width = 200
standard_height = 280

folder_names = ['background', 'border', 'decoration', 'dt', 'poster', 'product', 'text']

path = './'

for name in folder_names:
    imlist = glob.glob(path + 'data/' + name + '/*.png')
    for i in range(len(imlist)):
        im = Image.open(imlist[i])
        new_im = im.resize((standard_width, standard_height), Image.ANTIALIAS)
        new_im.save(path + 'data2/' + name + '/' + str(i) + '.png')


# In[31]:


# 生成负样本
import random

path1 = 'C:/Users/Mr-WH/Desktop/Auto Poster Generation/data2/'

backgroundlist = glob.glob(path + 'background/*.png')
borderlist = glob.glob(path + 'border/*.png')
decorationlist = glob.glob(path + 'decoration/*.png')
dtlist = glob.glob(path + 'dt/*.png')
posterlist = glob.glob(path + 'poster/*.png')
productlist = glob.glob(path + 'product/*.png')
textlist = glob.glob(path + 'text/*.png')

background_num = len(backgroundlist)
border_num = len(borderlist)
decoration_num = len(decorationlist)
dt_num = len(dtlist)
poster_num = len(posterlist)
product_num = len(productlist)
text_num = len(textlist)

# for each product
for i in range(product_num):
    
    im_product = Image.open(productlist[i])
    
    # produce 5 negative samples
    for j in range(5):
        
        # only one background
        im_background = Image.open(backgroundlist[random.randint(0, background_num - 1)])
        
        # decoration's number added randonmly
        for k in range(random.randint(0, 3)):
            # paste a decoration
            im_decoration = Image.open(decorationlist[random.randint(0, decoration_num - 1)])
            im_background.paste(im_decoration, (0, 0), im_decoration)
        
        # paste one border
        im_border = Image.open(borderlist[random.randint(0, border_num - 1)])
        im_background.paste(im_border, (0, 0), im_border)
        
        if random.randint(0, 2) > 1:
            im_dt = Image.open(dtlist[random.randint(0, dt_num - 1)])
            im_background.paste(im_dt, (0, 0), im_dt)
        
        # paste one product
        im_background.paste(im_product, (0, 0), im_product)
        
        # paste one text
        im_text = Image.open(textlist[random.randint(0, text_num - 1)])
        im_background.paste(im_text, (0, 0), im_text)
        
        im_background.save(path + 'poster_negative/n' + str(i * 5 + j) + '.png')

# im_background = Image.open(path + 'background/1.png')
# im_border = Image.open(path + 'border/0.png')
# im_decoration = Image.open(path + 'decoration/0.png')
# im_product = Image.open(path + 'product/0.png')
# im_text = Image.open(path + 'text/0.png')
# im_background.paste(im_decoration, (0, 0), im_decoration)
# im_background.paste(im_border, (0, 0), im_border)
# im_background.paste(im_product, (0, 0), im_product)
# im_background.paste(im_text, (0, 0), im_text)
# im_background.show()

