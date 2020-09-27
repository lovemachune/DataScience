import sys
import time
import re
import requests
import argparse
from bs4 import BeautifulSoup

PTT_URL = 'https://www.ptt.cc'

def check_format(url):
    res = requests.get(url, cookies={'over18': '1'})
    text = res.text.encode('utf-8')
    soup = BeautifulSoup(text, 'html.parser')
    divs = soup.find_all("span", class_ = "f2")
    for div in divs:
        if re.search("發信站", div.text) is not None:
            return True
    return False

def crawl(url):
    all_data = open('all_articles.txt', 'wb+')
    popular_data = open('all_popular.txt', 'wb+')
    temp_url = url
    month = 1
    flag = True
    start_flag = False
    while flag:
        res = requests.get(temp_url, cookies={'over18': '1'})
        text = res.text.encode('utf-8')
        soup = BeautifulSoup(text, 'html.parser')
        divs = soup.find_all("div", class_ = "r-ent")
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
            date = date_tag.string.split('/')
            date = int(date[0] + date[1])
            mydata = str(date) + ',' + title + ',' + PTT_URL + href + '\n'
            if date//100 == 1:
                start_flag = True
            if start_flag:
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

def push_process(url, users):
    res = requests.get(url, cookies={'over18': '1'})
    text = res.text.encode('utf-8')
    soup = BeautifulSoup(text, 'html.parser')
    pushs = soup.find_all('div', class_ = 'push')
    for push in pushs:
        push_span = push.find_all('span')
        tag = push_span[0].string
        user_name = push_span[1].string
        if user_name in users:
            if tag == '推 ':
                users[user_name]['like'] += 1
            elif tag == '噓 ':
                users[user_name]['boo'] += 1
        else:
            if tag == '推 ':
                users[user_name] = {'like':1, 'boo':0}
            elif tag == '噓 ':
                users[user_name] = {'like':0, 'boo':1}

def push(start_date, end_date):
    all_data = open('all_articles.txt', 'r', encoding='utf-8')
    push_data = open('push[%d-%d].txt' % (start_date, end_date), 'w+')
    datas = all_data.readlines()
    all_likes = 0
    all_boos = 0
    users = {}
    for data in datas:
        mydata = data.split(",")
        date = int(mydata[0])
        url = mydata[2].rstrip()
        if date > end_date:
            break
        if date >= start_date:
            push_process(url, users)  

    for _, tag in users.items():
        all_likes += tag['like']
        all_boos += tag['boo']

    push_data.write('all like: %d\n' % all_likes)
    push_data.write('all boo: %d\n' % all_boos)

    like_list = sorted(users, key=lambda x: (users[x]['like']*-1, x))
    boo_list = sorted(users, key=lambda x: (users[x]['boo']*-1, x))

    for index, name in enumerate(like_list[:10]):
        push_data.write('like #%d: %s %d\n' % (index+1, name, users[name]['like']))
    for index, name in enumerate(boo_list[:10]):
        push_data.write('boo #%d: %s %d\n' % (index+1, name, users[name]['boo']))
    
    all_data.close()
    push_data.close()

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
        # start of 2019
        start_time = time.time()
        print("start Beatuty 2019 crawling")
        url = "https://www.ptt.cc/bbs/Beauty/index2748.html"
        crawl(url)
        end = time.time()
        print("It cost %f sec" % (end - start_time))
    elif args.cmd[0] == 'push':
        start_time = time.time()
        print("start calculating pushing")
        push(int(args.cmd[1]), int(args.cmd[2]))
        end = time.time()
        print("It cost %f sec" % (end - start_time))
    elif args.cmd[0] == 'popular':
        popular(int(args.cmd[1]), int(args.cmd[2]))
    elif args.cmd[0] == 'keyword':
        keyword(args.cmd[1], int(args.cmd[2]), int(args.cmd[3]))
    else:
        print('Wrong input, please input again')
