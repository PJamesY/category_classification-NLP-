from bs4 import BeautifulSoup
import requests
import time
from multiprocessing import Pool, Manager
import pandas as pd
from functools import partial

end = 30
manager = Manager()
question_list = manager.list()

category_word = {
    "교육, 학문":11,
    "컴퓨터 통신":1,
    "게임":2,
    "엔터테인먼트, 예술":3,
    "생활":8,
    "건강":7,
    "사회정치":6,
    "경제":4,
    "여행":9,
    "스포츠 레저":10,
    "쇼핑":5,
}

def make_data(category,question_ls):
    category_list = [category for _ in range(len(question_ls))]
    data = {'question': question_list, 'category': category_list}
    return data

def question_to_list(category, start):

    global question_list

    url = 'https://kin.naver.com/qna/list.nhn?dirId={}&queryTime=2019-04-03%2000%3A05%3A58&page={}'.format(category,start)
    req = requests.get(url)

    if req.ok:
        html = req.text
        soup = BeautifulSoup(html, 'html.parser')

        questions = soup.select(
            '#au_board_list > tr > td  > a'
        )

        for question in questions:
            url1 = 'https://kin.naver.com{}'.format(question['href'])
            req1 = requests.get(url1)

            if req1.ok:
                html1 = req1.text
                soup1 = BeautifulSoup(html1, 'html.parser')
                titles = soup1.select('#content > .question-content > .question-content__inner > .c-heading > .c-heading__title > .c-heading__title-inner > .title')

                # list에 넣어준다
                for title in titles:
                    question_list.append(title.text)

def crawling():
    df = pd.DataFrame([])
    pool = Pool(processes=4)
    for category, _ in category_word.items():
        func = partial(question_to_list, category)
        pool.map(func, range(1,end,10))
        data = make_data(category, question_list)
        df = df.append(pd.DataFrame(data), ignore_index = True)

    df.to_csv('data.csv')
    return df



if __name__ == '__main__':
    crawling()

