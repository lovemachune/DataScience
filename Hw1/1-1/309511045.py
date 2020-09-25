import sys
import time
import re
import requests
import argparse
from bs4 import BeautifulSoup

PTT_URL = 'https://www.ptt.cc'

def crawl(url):
    all_data = open('all_articles.txt', 'wb+')
    popular_data = open('all_popular.txt', 'wb+')
    temp_url = url
    month = 1
    flag = True
    while flag:
        res = requests.get(temp_url, cookies={'over18': '1'})
        text = res.text.encode('utf-8')
        soup = BeautifulSoup(text, 'html.parser')
        divs = soup.find_all("div", "r-ent")
        temp_url = PTT_URL + soup.find_all(class_ = 'btn wide')[2]['href']
        print(temp_url)
        for div in divs:
            title_tag = div.find(class_ = 'title').find('a')
            if title_tag == None:
                continue
            title = title_tag.string
            href = title_tag['href']
            date_tag = div.find(class_ = 'date')
            rate = div.find(class_ = 'nrec').string
            date = date_tag.string.split('/')
            date = int(date[0] + date[1])
            mydata = str(date) + ',' + title + ',' + PTT_URL + href + '\n'
            if date//100 != month:
                month += 1
            if month == 13:
                flag = False
                break
            all_data.write(mydata.encode('utf-8'))
            if rate == "爆":
                popular_data.write(mydata.encode('utf-8'))
    all_data.close()
    popular_data.close()
def push(start_date, end_date):
    print(start_date)
    print(end_date)

def popular(start_date, end_date):
    print(start_date)
    print(end_date)

def keyword(key, start_date, end_date):
    print(start_date)
    print(end_date)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PTT crawling')
    parser.add_argument('cmd', type=str, nargs='+', help='PTT crawling')
    args = parser.parse_args()

    if args.cmd[0] == 'crawl':
        start_time = time.time()
        print("crawling")
        # start of 2019
        url = "https://www.ptt.cc/bbs/Beauty/index2749.html"
        crawl(url)
        end = time.time()
        print("It cost %f sec" % (end - start_time))
    elif args.cmd[0] == 'push':
        print("pushing")
        push(int(args.cmd[1]), int(args.cmd[2]))
    elif args.cmd[0] == 'popular':
        popular(int(args.cmd[1]), int(args.cmd[2]))
    elif args.cmd[0] == 'keyword':
        keyword(args.cmd[1], int(args.cmd[2]), int(args.cmd[3]))
    else:
        print('Wrong input, please input again')
