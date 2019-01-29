#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[81]:



import requests

def setProxy():
#     proxy = {
#         "http": "http://root:1M_qRK_]q4Ko8n6g@45.76.245.86:443/",
#     }
    proxy = {
        "http": "http://45.76.245.86:8080",
    }
#     a = requests.get("https://see.tongji.edu.cn", proxies=proxy)
#     a = requests.get("https://www.cmu.edu/", proxies=proxy)
    a = requests.get("https://www.pinterest.com/", proxies=proxy)


    print(a.text.encode('latin').decode('utf-8'))

setProxy()


# In[75]:


import urllib.request as request
import requests

proxies = {
    'https': 'https://45.76.245.86:8080',
    'http': 'http://45.76.245.86:8080'
}
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'
}

print('--------------使用urllib--------------')
google_url = 'https://www.google.com'
opener = request.build_opener(request.ProxyHandler(proxies))
request.install_opener(opener)

req = request.Request(google_url, headers=headers)
response = request.urlopen(req)

print(response.read().decode())

print('--------------使用requests--------------')
response = requests.get(google_url, proxies=proxies)
print(response.text)


# In[79]:


proxy = {
    'http': 'socks5://root:1M_qRK_]q4Ko8n6g@45.76.245.86:443',
    'https': 'socks5://root:1M_qRK_]q4Ko8n6g@45.76.245.86:443',
}
header = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'
}

# 请求这样写
# resp = requests.get("https://www.google.com/", proxies=proxy, headers=header)
resp = requests.get("https://www.google.com/", proxies=proxy)


# In[15]:


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
        
    


# In[83]:


# -*- coding: utf-8 -*-

import json
from lxml import etree
import requests
import click

headers = {
    "Origin": "https://www.instagram.com/",
    "Referer": "https://www.instagram.com/_8_jjini/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/58.0.3029.110 Safari/537.36",
    "Host": "www.instagram.com",
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "accept-encoding": "gzip, deflate, sdch, br",
    "accept-language": "zh-CN,zh;q=0.8",
    "X-Instragram-AJAX": "1",
    "X-Requested-With": "XMLHttpRequest",
    "Upgrade-Insecure-Requests": "1",
}

BASE_URL = "https://www.instagram.com/_8_jjini/"

proxy = {
    'http': 'http://127.0.0.1:38251',
    'https': 'http://127.0.0.1:38251'
}


def crawl():
    click.echo('start')
    try:
        res = requests.get(BASE_URL, headers=headers, proxies=proxy)
        html = etree.HTML(res.content.decode())
        all_a_tags = html.xpath('//script[@type="text/javascript"]/text()')
        for a_tag in all_a_tags:
            if a_tag.strip().startswith('window'):
                data = a_tag.split('= {')[1][:-1]  # 获取json数据块
                js_data = json.loads('{' + data, encoding='utf-8')
                edges = js_data["entry_data"]["ProfilePage"][0]["graphql"]["user"]["edge_owner_to_timeline_media"]["edges"]
                for edge in edges:
                    if top_url and top_url == edge["node"]["display_url"]:
                        in_top_url_flag = True
                        break
                    click.echo(edge["node"]["display_url"])
                    new_imgs_url.append(edge["node"]["display_url"])
                click.echo('ok')
    except Exception as e:
        raise e


if __name__ == '__main__':
    crawl()

