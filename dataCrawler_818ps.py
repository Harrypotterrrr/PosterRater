#!/usr/bin/env python
# coding: utf-8

# In[1]:


from urllib import request
from bs4 import BeautifulSoup
import sys
import os
print(os.getcwd())

debug = False
url = r"https://818ps.com/muban/0-0-0-0-1661-null-0-0-0/"
img_path = r"./data/818ps/"
http_head = r"http:"
img_ctr = 0

if not os.path.isdir(img_path):
    os.makedirs(img_path)


def setProxy():
    proxy = {
        'http': '102.60.17.52',
        'https': '102.60.17.53'
    }
    proxies = request.ProxyHandler(proxy)
    opener = request.build_opener(proxies)
    return opener

def getHtml(url):
    
    page = request.urlopen(url)
    html = page.read()
    
    # convert byte to str
    return html.decode()

def getContent(html):
    
    div1 = "<div style=\"margin-top: 80px;\">"
    html = html.partition(div1)[2]
    
    div2 = "<div class=\"content banxin\" style=\"text-align: center;\">"
    html = html.partition(div2)[2]
    
    div3 = "<input type=\"hidden\" id=\"phoneBindNeedDownload\">"
    html = html.partition(div3)[0]
    
    return html

def save_img():
    global img_ctr
    
    if debug:
        assert len(img_url) == len(img_title)

    l_list = len(img_url)
    for i in range(l_list):
        
        title = img_title[i] + '.jpg'
        request.urlretrieve(img_url[i], img_path+title)
        
        print("Save successully [" + str(img_ctr) + "] :" + img_path+title)
        img_ctr += 1


# In[2]:


if __name__ == "__main__":
    
    seq_url = 1
    # the max_index of url page is uncertained.
    try:

        while True:
            img_url = []
            img_title = []
            
            if debug:
                print("seq_url: ", seq_url)
            
            cur_url = url + str(seq_url) + ".html"
            html = getContent(getHtml(cur_url))
    
#             if debug:
#                 with open("output.txt", 'w') as file:
#                     file.write(html)

            bs = BeautifulSoup(html, 'html.parser')
            blocks = bs.select('#masonry')[0].select('.box')

            for block in blocks:

                # Generally, label '<dev>' isn't contained by label '<a>' = = 
                block = block.select('.min-img')[0].select('.lazy')[0]
                
                if block.has_attr('src') == True:
                    if debug:
                        print(block['title'])
                        print(block['src'])

                    img_title.append(block['title'])
                    img_url.append(http_head + block['src'])
                    
                else:
                    if debug:
                        print(block['title'])
                        print(block['data-original'])

                    img_title.append(block['title'])
                    img_url.append(http_head + block['data-original'])
            
            seq_url += 1
            save_img()
        
    except ValueError:
        print(Exception.message)

