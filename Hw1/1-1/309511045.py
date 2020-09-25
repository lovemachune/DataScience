import sys
import re
import requests
import argparse
from bs4 import BeautifulSoup

PTT_URL = 'https://www.ptt.cc'

def crawl(url):
    all_data = open('all_articles.txt', 'wb+')
    popular_data = open('all_popular.txt', 'wb+')
    temp_url = url
    res = requests.get(temp_url, cookies={'over18': '1'})
    text = res.text.encode('utf-8')
    soup = BeautifulSoup(text, 'html.parser')
    divs = soup.find_all("div", "r-ent")
    for div in divs:
        date_tag = div.find(class_ = 'date')
        title_tag = div.find(class_ = 'title')
        href = title_tag.a['href']
        title = title_tag.a.string
        rate = div.find(class_ = 'nrec').string
        date = date_tag.string.split('/')
        date = int(date[0] + date[1])
        mydata = str(date) + ',' + title + ',' + PTT_URL+href + '\n'
        all_data.write(mydata.encode('utf-8'))

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
        print("crawling")
        # start of 2019
        url = "https://www.ptt.cc/bbs/Beauty/index2749.html"
        crawl(url)
    elif args.cmd[0] == 'push':
        print("pushing")
        push(int(args.cmd[1]), int(args.cmd[2]))
    elif args.cmd[0] == 'popular':
        popular(int(args.cmd[1]), int(args.cmd[2]))
    elif args.cmd[0] == 'keyword':
        keyword(args.cmd[1], int(args.cmd[2]), int(args.cmd[3]))
    else:
        print('Wrong input, please input again')
