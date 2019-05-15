import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from preprocessing import ArticlePreProcessing


# dataframe 가져오기(load)
# preprocessing.py에서 전처리하고 저장한 데이터를 make_cnn_dataset.py로 불러온다
df = pd.read_csv('data/article.csv')


class MakeCnnDataset:

    def __init__(self, konlpy="Okt", max_num_words=15000):
        """
        :param konlpy: 형태소 분석기 선택 ['Okt', 'Komoran']
        :param max_num_words: 최대 사용할 단어의 갯
        """
        self.konlpy = konlpy
        self.max_num_words = max_num_words

        # self.docs : string sentence - > ['sentence1','sentence2','...']
        if self.konlpy == "Okt":
            preprocessing = ArticlePreProcessing(df, remove=False)
            self.df = preprocessing.doc2sentence()
            self.docs = np.array(self.df['joined_sentence'])
        else:
            preprocessing = ArticlePreProcessing(df, konlpy='Komoran', remove=False)
            self.df = preprocessing.doc2sentence()
            self.docs = np.array(self.df['joined_sentence'])

    def sentence_length(self):
        """
        :return: max length of sentence
        """
        max_length = 0

        for word_tag_ls in self.df['word_tagging']:
            if len(word_tag_ls) >= max_length:
                max_length = len(word_tag_ls)

        return max_length

    def make_data(self):
        # tokenizer
        tokenizer = Tokenizer(num_words=self.max_num_words)
        tokenizer.fit_on_texts(self.docs)
        sequences = tokenizer.texts_to_sequences(self.docs)

        word_to_index = tokenizer.word_index
        vocabulary = word_to_index.keys()

        sentence = [len(sentence.split()) for sentence in self.docs]
        max_length = self.sentence_length()

        print('Text informations:')
        print('sentence max length: %i / min length: %i / mean length: %i / limit length: %i' % (np.max(sentence),
                                                                                                 np.min(sentence),
                                                                                                 np.mean(sentence),
                                                                                                 max_length))
        print('총 단어 갯수: %i / 한 문장 속에 제한: %i' % (len(word_to_index), self.max_num_words))

        # make X data set
        # Padding all sequences to same length of `MAX_SEQ_LENGTH`, 최대 문장 길이보다 작은 문장은 나머지 단어들은 padding 0으로 처리 된다
        X = pad_sequences(sequences, maxlen=max_length, padding='post')

        # X
        # array([[ 2587,    36,   969, ...,     0,     0,     0],
        #       [ 1575,     4,   121, ...,     0,     0,     0],
        #       [    5,     2,  2639, ...,     0,     0,     0],
        #       ...,
        #       [  742,  4671, 14415, ...,     0,     0,     0],
        #       [11686,    61, 14415, ...,     0,     0,     0],
        #       [14359,   741,     1, ...,     0,     0,     0]], dtype=int32)

        # make y data set
        cateory_to_label = {'education': 0,
                            'health': 1,
                            'politics': 2,
                            'technology': 3,
                            'it': 3,
                            'sports': 4,
                            'entertainment': 5,
                            'travel': 6,
                            }

        # dataframe에 카테고리들을 int label 컬럼을 추가시킨다
        self.df['category2label'] = self.df['category'].apply(lambda category: cateory_to_label[category])

        y = to_categorical(list(self.df['category2label']))

        # y
        # array([[0., 0., 0., ..., 0., 1., 0.],
        #        [1., 0., 0., ..., 0., 0., 0.],
        #        [0., 1., 0., ..., 0., 0., 0.],
        #        ...,
        #        [0., 0., 1., ..., 0., 0., 0.],
        #        [1., 0., 0., ..., 0., 0., 0.],
        #        [0., 0., 1., ..., 0., 0., 0.]], dtype=float32)
        return X, y, word_to_index, vocabulary


if __name__ == "__main__":
    make_cnn_dataset = MakeCnnDataset("Komoran")
    X, y, word2index, voca = make_cnn_dataset.make_data()
    print(X.shape)




