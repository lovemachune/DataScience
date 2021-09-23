import sys
import time
import re
import requests
import argparse
import numpy as np
#from bs4 import BeautifulSoup
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dense

PTT_URL = 'https://www.ptt.cc'

def check_format(soup):
    divs = soup.find_all("span", class_ = "f2")
    for div in divs:
        if re.search("發信站", div.text) is not None:
            return True
    return False

def connect(url):
    time.sleep(0.05)
    res = requests.get(url, cookies={'over18': '1'})
    text = res.text.encode('utf-8')
    soup = BeautifulSoup(text, 'html.parser')
    if check_format(soup) is False:
        return None
    return soup

def crawl(url):
    all_data = open('all_articles.txt', 'wb+')
    beauty_data = open('beauty.txt', 'wb+')
    ugly_data = open('ugly.txt', 'wb+')
    temp_url = url
    month = 1
    flag = True
    start_flag = False
    while flag:
        time.sleep(0.1)
        res = requests.get(temp_url, cookies={'over18': '1'})
        text = res.text.encode('utf-8')
        soup = BeautifulSoup(text, 'html.parser')
        divs = soup.find_all("div", class_ = "r-ent")
        if start_flag:
            print(temp_url)
        temp_url = PTT_URL + soup.find_all(class_ = 'btn wide')[2]['href']
        for div in divs:
            title_tag = div.find(class_ = 'title').find('a')
            if title_tag == None:
                continue
            title = title_tag.string
            if re.search("公告", title):
                continue
            href = title_tag['href']
            date_tag = div.find(class_ = 'date')
            rate = div.find(class_ = 'nrec').string
            if rate == '爆':
                rate = 100
            elif rate is not None and rate.isdigit():
                rate = int(rate)
            else:
                rate = 0
            date = date_tag.string.split('/')
            date = int(date[0] + date[1])
            mydata = str(date) + ',' + title + ',' + PTT_URL + href + '\n'
            if date//100 == 1:#re.search('雲林斗六跨年晚會', title) is not None:
                start_flag = True
            if start_flag:
                if date//100 != month:
                    month += 1
                if month == 13:
                    flag = False
                    break
                all_data.write(mydata.encode('utf-8'))
                #print(rate)
                if rate >= 35:
                    beauty_data.write(mydata.encode('utf-8'))
                else:
                    ugly_data.write(mydata.encode('utf-8'))

    all_data.close()
    beauty_data.close()
    ugly_data.close()

def save_image(addr, url, index):
    r = requests.get(url, allow_redirects=True)
    open(addr + str(index)+'.jpg', 'wb').write(r.content)

def beauty():
    beauty_data = open('beauty.txt', 'r', encoding='utf-8')
    datas = beauty_data.readlines()
    index = 0
    addr = './pics/beauty/'
    pattern = '^http.*(png|jpe?g)$'
    for data in datas:
        mydata = data.split(",")
        url = mydata[-1].rstrip()
        soup = connect(url)
        if soup is None:
            print("NO 發信站 : %s" % url)
            continue
        contents = soup.find_all('a')
        print(url)
        for content in contents:
            href = content['href']
            if re.match(pattern, href) is not None:
                save_image(addr, href, index)
                index += 1

def ugly():
    ugly_data = open('ugly.txt', 'r', encoding='utf-8')
    datas = ugly_data.readlines()
    index = 0
    addr = './pics/ugly/'
    pattern = '^http.*(png|jpe?g)$'
    for data in datas:
        mydata = data.split(",")
        url = mydata[-1].rstrip()
        soup = connect(url)
        if soup is None:
            print("NO 發信站 : %s" % url)
            continue
        contents = soup.find_all('a')
        print(url)
        for content in contents:
            href = content['href']
            if re.match(pattern, href) is not None:
                save_image(addr, href, index)
                index += 1

def predict_push():
    img_height = 180
    img_width = 180
    model = tf.keras.models.load_model('./my_model')
    myfile = open('imgfilelistname.txt', 'r', encoding='utf-8')
    output = open('classification.txt', 'w+')
    datas = myfile.readlines()
    imgs = []
    for data in datas:
        img = cv2.imread(data.rstrip())
        img = cv2.resize(img, (img_height, img_width))
        imgs.append(img)
    imgs = np.array(imgs)
    predicts = model.predict(imgs)
    for i, j in predicts:
        if(i>j): #beauty
            output.write('1')
        else:
            output.write('0')
    output.close()
    myfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PTT crawling')
    parser.add_argument('cmd', type=str, nargs='+', help='PTT crawling')
    args = parser.parse_args()

    if args.cmd[0] == 'crawl':
        # start of 2019
        start_time = time.time()
        print("Beatuty 2019 crawling")
        url = "https://www.ptt.cc/bbs/Beauty/index2745.html"
        crawl(url)
        end = time.time()
        print("It cost %f sec" % (end - start_time))
    elif args.cmd[0] == 'pic':
        start_time = time.time()
        print("Crawl images")
        #beauty()
        #ugly()
        end = time.time()
        print("It cost %f sec" % (end - start_time))
    elif args.cmd[0] == 'imgfilelistname':
        start_time = time.time()
        predict_push()
        end = time.time()
        print("It cost %f sec" % (end - start_time))
    else:
        print('Wrong input, please input again')
