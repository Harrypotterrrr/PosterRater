#!/usr/bin/env python
# coding: utf-8

# In[1]:


from urllib import request
from bs4 import BeautifulSoup
import sys
import os
print(os.getcwd())

debug = False
url = r"https://www.pinterest.com/search/pins/?q=poster%20food&rs=typed&term_meta[]=poster%7Ctyped&term_meta[]=food%7Ctyped"
img_path = r"./data/pinterest/"
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
    
    div1 = "<div class=\"vbI XiG\" style=\"height: 4892px; width: 520px;\">"
    html = html.partition(div1)[2]
    
    div2 = "<div class=\"vbI XiG\" style=\"width: 520px;\">"
    html = html.partition(div2)[0]
    
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


# In[ ]:


import requests

# 代理服务器
proxyHost = "b5.t.16yun.cn"
proxyPort = "6460"

# 代理隧道验证信息
proxyUser = "16OJOKVZ"
proxyPass = "822007"

proxyMeta = "http://%(user)s:%(pass)s@%(host)s:%(port)s" % {
    "host": proxyHost,
    "port": proxyPort,
    "user": proxyUser,
    "pass": proxyPass,
}

# 设置 http和https访问都是用HTTP代理
proxies = {
    "http": proxyMeta,
    "https": proxyMeta,
}


proxy = { "https": "45.76.245.86:443" }
header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Accept-Language': 'zh-CN,zh;q=0.8'}
response = requests.get(url, headers=header, proxies=proxy)

print(response.text)
# html = getContent(getHtml(url))


# In[4]:


# 代理服务器
proxyHost = "b5.t.16yun.cn"
proxyPort = "6460"

# 代理隧道验证信息
proxyUser = "16OJOKVZ"
proxyPass = "822007"

proxyMeta = "http://%(user)s:%(pass)s@%(host)s:%(port)s" % {
    "host": proxyHost,
    "port": proxyPort,
    "user": proxyUser,
    "pass": proxyPass,
}

# 设置 http和https访问都是用HTTP代理
proxies = {
    "http": proxyMeta,
    "https": proxyMeta,
}

print(proxies["http"])


# In[ ]:


if __name__ == "__main__":
    
    img_url = []
    img_title = []


    html = getContent(getHtml(url))


    bs = BeautifulSoup(html, 'html.parser')
    blocks = bs.select('.Yl-')

    for block in blocks:

        block = block.find_all(class = 'XiG zI7 iyn Hsu')
        srcSet = block['srcset'].split(',')

        print(len(stcSet), srcSet)
        src = srcSet[-1]
        
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
        
    

