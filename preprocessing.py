from konlpy.tag import Okt, Kkma, Komoran
import pandas as pd
import csv
import itertools

article_df = pd.read_csv('data/article.csv')


def extract_meaning_word(row):
    """
    조사와 감탄사등 필요없는 토큰은 제거해주고 제거해준 list를 반환해주는 함수

    :param row: [(word1,tag1), (word2,tag2),...] 형태의 row list(dataframe)
    :return: 품사 tagging한 컬럼에서 조사, 어미, 감탄사, 이모티콘, punctuation 을 제거해준 list
    """
    meaning_word_tagging_ls = []
    # Noun : 명사 / Verb : 동사 / Adjective : 형용사
    # NNP : 고유명사 / NNG : 보통 명사 / VV : 동사 / XR : 어근 / VA : 형용사 /
    meaning_tag_ls = ["Noun","Verb","Adjective",'NNG', 'NNP', 'VV', 'XR','VA']

    for word in row:
        if word[1] in meaning_tag_ls:
            meaning_word_tagging_ls.append(word)

    return meaning_word_tagging_ls


def remove_stopwords(row):
    """
    불용어를 제거해준 list 반환해주는 함수

    :param row: [(word1,tag1), (word2,tag2),...] 형태의 row list(dataframe)
    :return: stopwords를 제거한 품사 tagging list 반환
    """
    # stopword 외부에서 가져온것
    file = open('data/stopwords-ko.txt', 'r')
    reader = csv.reader(file)
    all_rows = [row for row in reader]

    # flatten list
    stopwords = list(set(itertools.chain(*all_rows)))

    no_stopwords_ls = []
    for word in row:
        if word[0] not in stopwords:
            no_stopwords_ls.append(word)

    return no_stopwords_ls


def word_join(row):
    """
    word1, word2 등을 한칸의 띄어쓰기로 join 하는 함수
    :param row: [(word1,tag1), (word2,tag2),...] 형태의 row list(dataframe)
    :return: string sentence
    """

    sentence = ''

    for word in row:
        sentence += word[0] + ' '
    return sentence.strip()


class PreProcessing:

    def __init__(self, df=article_df, konlpy=Okt, model="LDA"):
        """
        :param df: 전처리를 할 데이터 프레임
        :param konlpy: 형태소 분석기 선택 [Okt, Kkma, Komoran]
        :param model: 사용할 머신러닝 / 딥러닝 모델 [CNN, RNN, LDA, word2vec]

        """
        self.df = df
        self.konlpy = konlpy
        self.model = model

    def change_data_type(self, type=str):
        """
        dataframe의 data type을 바꿔준다

        :param type: 바꾸고 싶은 type을 적는다
        :return: 데이터 타입이 수정된 dataframe
        """
        self.df = self.df.astype(type)

        return self.df

    def drop_duplicate(self, *col):
        """
        중복되는 데이터는 지워준다
        :param *col: 중복된 데이터들을 제거하는데 고려하는 columns
        DataFrame with duplicate rows removed, optionally only considering certain columns.
        """
        # 행간의 중복 제거
        self.df.drop_duplicates(subset=list(*col), inplace=True)

    def text_pre_processing(self, column):
        """

        :param column: text 전처리 하고자 하는 column name
        :return: 특정 column의 text 전처리가 완료된 dataframe
        """
        # 이름, 고유명사는 하나로 통일
        self.df[column] = self.df[column].replace('문 정권', '문재인 정권')
        self.df[column] = self.df[column].replace('문 정부', '문재인 정부')
        self.df[column] = self.df[column].replace('문 대통령', '문재인 대통령')
        self.df[column] = self.df[column].replace('도널드 트럼프', '트럼프')
        self.df[column] = self.df[column].replace('KT', '케이티')
        self.df[column] = self.df[column].replace('SK', '에스케이')

        self.df[column] = self.df[column].replace('승', '승리')
        self.df[column] = self.df[column].replace('TV', '티프이')

        # 한·미 같은 단어는 한국 미국으로 나타내었다
        self.df[column] = self.df[column].replace("한[-·]미", '한국 미국')
        self.df[column] = self.df[column].replace("한미", '한국 미국')
        self.df[column] = self.df[column].replace("북[-·]미", '북한 미국')
        self.df[column] = self.df[column].replace("북미", '북한 미국')

        # 시간, 날짜, 수치 등은 모두 제거(10월, 4일, 수요일, 4시, 40%)
        self.df[column] = self.df[column].replace('[0-9]{1,10}[가-힣\%]{1,10}', ' ', regex=True)
        self.df[column] = self.df[column].replace('[가-힣]요일', ' ', regex=True)

        # [영상], [포토] 등 단어 제거
        self.df[column] = self.df[column].replace("\[.{1,20}\]", ' ', regex=True)

        # 뒤에 기자이름은 없앤다
        self.df[column] = self.df[column].replace('[가-힣]{1,4} 기자', ' ', regex=True)
        self.df[column] = self.df[column].replace('기자', ' ', regex=True)

        # 한글만 텍스트로 남긴다
        self.df[column] = self.df[column].replace('[^가-힣\s]', ' ', regex=True)

        # multiple space -> single space
        self.df[column] = self.df[column].replace('\s+', ' ', regex=True)

        # 양쪽 끝 white space를 delete
        self.df[self.df.columns] = self.df.apply(lambda x: x.str.strip())

        self.drop_duplicate(column)
        return self.df

    def morpheme_tagging(self, column):
        """
        konlpy에 있는 것 중에 한가지로 형태소 분석 tagging을 합니다

        :param: 형태소 품사 tagging을 하려는 column (type:str)
        :return: tagging 분리 column이 추가된 dataframe
        """
        if self.konlpy == "Okt":
            konlpy = Okt()
        elif self.konlpy == "Kkma":
            konlpy = Kkma()
        elif self.konlpy == "Komoran":
            konlpy = Komoran()
        else:
            raise ValueError('select in Okt, Kkma, Komoran')

        self.df['word_tagging'] = self.df[column].apply(lambda sentence:konlpy.pos(sentence))

        return self.df

    def pre_processing(self, column):
        """

        :param column: 전처리하고자 하는 column (type : str)
        :return: 전처리된 dataframe
        """

        self.text_preprocessing(column)

        self.morpheme_tagging(column)

        self.df['word_tagging'] = self.df['word_tagging'].apply(lambda row: extract_meaning_word(row))
        self.df['word_tagging'] = self.df['word_tagging'].apply(lambda row: remove_stopwords(row))
        self.df['joined_sentence'] = self.df['word_tagging'].apply(lambda row: word_join(row))

        self.df = self.df[['word_tagging', 'joined_sentence', 'category']]

        self.df.reset_index(inplace=True)
        return self.df

    def title_plus_article(self):
        """
        LDA topic modeling을 위해서 사용하는 dataframe
        title column과 article column을 통합시킨 title_article column을 생성
        :return: Dataframe (column : title_article, category)
        """
        self.change_data_type(str)
        self.drop_duplicate('title', 'article')

        # title과 article을 합쳐주는 column을 만든다
        self.df["title_article"] = self.df['title'] + self.df['article']
        self.df = self.df[['title_article', 'category']]

        self.preprocessing('title_article')

    def doc2sentence(self):
        """
        CNN, word2vec 모델에 사용
        document article을 문장으로 다 쪼개서 문장 하나하나를 하나의 데이터로 받는다
        :return:
        """
        self.change_data_type(str)
        self.drop_duplicate('title', 'article')

        self.df = pd.concat([pd.DataFrame({'article': doc, 'category': row['category']}, index=[0])
                             for _, row in self.df.iterrows()
                             for doc in row['article'].split('.') if doc != ''])

        self.df = self.df[['article', 'category']]

        self.preprocessing('article')

