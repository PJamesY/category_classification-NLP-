import urllib.request
from bs4 import BeautifulSoup
from multiprocessing import Pool, Manager
import pandas as pd
from functools import partial

category_word = {
    'education': 'society/schooling',
    'health': 'society/health',
    'politics': 'politics/politics_general',
    'technology': 'science/technology',
    'it': 'economy/it',
    'sports': 'sports/sports_general',
    'entertainment': 'culture/entertainment',
    'travel': 'culture/travel',
}

# 끝 페이지
end = 5000
manager = Manager()
# 기사의 제목을 담을 list
title_list = manager.list()
# 기사의 content를 담을 list
article_list = manager.list()


def make_dic_data(category,article_ls,title_ls):
    """
    :param category: 기사가 속한 category
    :param article_ls: 기사의 content list
    :param title_ls: 기사의 title list
    :return: dictionary type의 데이터 반환
    """
    category_list = [category for _ in range(len(title_ls))]
    data = {'title': title_ls,
            'article': article_ls,
            'category': category_list,}
    return data


def question_to_list(category, page):
    """
    :param category: 기사가 속한 category
    :param page: 크롤링할 페이지
    """
    global article_list
    global title_list

    url = "http://www.hani.co.kr/arti/{}/list{}.html".format(category, page)
    response = urllib.request.urlopen(url)
    soup = BeautifulSoup(response, 'html.parser')
    results = soup.select('#section-left-scroll-in > .section-list-area > div .article-title a')

    for result in results:
        url_article = "http://www.hani.co.kr{}".format(result.attrs["href"])
        response = urllib.request.urlopen(url_article)
        soup_article = BeautifulSoup(response, "html.parser")

        # 기사 제목
        title = soup_article.select_one("#article_view_headline > h4 > span").text
        # 기사의 글
        contents = soup_article.select_one("#a-left-scroll-in > .article-text > .article-text-font-size > .text")

        # 기사 글 가공 처리
        article = ''
        try:
            for content in contents.contents:
                stripped = str(content).strip()
                if stripped == "":
                    continue
                if stripped[0] not in ['<', '/']:
                    article += str(content).strip()
        except AttributeError:
            print('Pass')

        # 각 기사의 제목과 글 list에 append
        title_list.append(title)
        article_list.append(article)


def crawling():
    global article_list
    global title_list

    df = pd.DataFrame([])
    # pool = Pool(processes=4)
    for category, _ in category_word.items():
        pool = Pool(processes=4)
        func = partial(question_to_list, category_word[category])
        pool.map(func, range(1,end,10))
        data = make_dic_data(category, article_list, title_list)
        df = df.append(pd.DataFrame(data), ignore_index = True)

        # content 다시 초기화
        title_list = manager.list()
        article_list = manager.list()

    df.to_csv('article11.csv')


if __name__ == '__main__':
    crawling()
