# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim import models
from random import shuffle
import time
import os
import pandas as pd
import numpy as np

DATA_FRAME = pd.read_pickle('data/komoran_LDA_dataset.pkl')

# article_data = ["doc1", "doc2", ...]
ARTICLE_DATA = DATA_FRAME['joined_sentence']

# categories = array(['education', 'health', 'politics', 'technology', 'it', 'sports', 'entertainment', 'travel'], dtype=object)
CATEGORIES = DATA_FRAME.category.unique()

# topic category 의 갯수 = 8개
N_TOPICS = len(CATEGORIES)

# documnets 의 갯수 = 7700
N_DOCUMENTS = len(DATA_FRAME)

# Count vectorizer 사용
tf_vectorizer = CountVectorizer(ngram_range=(1,1), min_df=5).fit(ARTICLE_DATA)

# words 갯수 = 18656
WORDS = tf_vectorizer.get_feature_names()

# tf_docs.shape = (7700, 18656)
TF_DOCUMENTS = tf_vectorizer.transform(ARTICLE_DATA)

# LDA를 돌리고 난뒤 임의로 생긴 토픽들을 이름을 지정해주는 상수
# LDA_TOPIC_NAME의 길이는 N_TOPIC과 같을 필요는 없다
LDA_TOPIC_NAME = ['life/health','education1','health','it/tech','entertainment','sports','politics','education2']


def show_top_words(n_top_words=20, n_lda_topics=N_TOPICS):
    """
        show dominant words in each topic
        topic1, topic2, topi3 가 어떤 topic 인지 확인 해서 Topic2Vec의 lda_topic_name을 바꿔준다

        :param n_top_words: 단어를 몇개 까지 뽑을 것인지 정하는 파라미터
        :param n_lda_topics: LDA 상에서 원하는 topic의 갯수
        :return: {topic_1:[word1, word2, ..], topic_2:[word3, word4, ...]} 형태의 딕셔너리 타입
        """
    lda = LatentDirichletAllocation(n_components=n_lda_topics, max_iter=5, learning_method='online', learning_offset=60, random_state=0, n_jobs=-1)
    lda.fit(TF_DOCUMENTS)

    topic_word_dict = {}
    for topic_idx, topic in enumerate(lda.components_):
        topic_word_dict["topic_" + str(topic_idx)] = [WORDS[i] for i in topic.argsort()[:-n_top_words - 1:-1]]

    return topic_word_dict


def dominant_topic(n_lda_topics=N_TOPICS, lda_topic_name=LDA_TOPIC_NAME):
    """
    show the dominant topic in each document
    :param n_lda_topics: LDA 상에서 원하는 topic의 갯수
    :param lda_topic_name: LDA 상에서 나타는 topic의 이름을 변경한 list
    :return: dataframe
    """
    if n_lda_topics != len(lda_topic_name):
        raise Exception("The number of LDA Topic title must be number of LDA Topic")

    lda = LatentDirichletAllocation(n_components=n_lda_topics, max_iter=5, learning_method='online', learning_offset=60, random_state=0, n_jobs=-1)
    lda.fit(TF_DOCUMENTS)
    # Create Document - Topic Matrix
    lda_output = lda.transform(TF_DOCUMENTS)

    # column names
    lda_topic_names = lda_topic_name

    # index names
    doc_names = ["Doc_" + str(i) for i in range(len(N_DOCUMENTS))]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=lda_topic_names, index=doc_names)

    # Get dominant topic for each document
    dominant_topic_idx = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = [(lambda idx:lda_topic_name[idx])(idx) for idx in dominant_topic_idx]
    return df_document_topic


def word2topic(n_lda_topics=N_TOPICS, lda_topic_name=LDA_TOPIC_NAME):
    """
    show the dominant topic in each word
    :param n_lda_topics: LDA 상에서 원하는 topic의 갯수
    :param lda_topic_name: LDA 상에서 나타는 topic의 이름을 변경한 list
    :return: dictionary 형태의 key(word)-value(topic)
    """
    lda = LatentDirichletAllocation(n_components=n_lda_topics, max_iter=5, learning_method='online', learning_offset=60, random_state=0, n_jobs=-1)
    lda.fit(TF_DOCUMENTS)

    per_topic_dist_lda = lda.components_

    # dominant topic in each word
    most_topic_idx = np.argmax(per_topic_dist_lda, axis=0)
    most_topic_name = [(lambda idx:lda_topic_name[idx])(idx) for idx in most_topic_idx]
    word_and_topic = zip(WORDS, most_topic_name)
    word2topic_dict = {word: topic for word, topic in word_and_topic}

    return word2topic_dict

def map_doc_to_topic(tokenized_text, category, doc_id_number, word2topic_dict):
    """
    각 document 마다 맨 처음에 document가 속한 category가 있고 그 다음부터는 단어의 dominant topic이 나열된 list
    :param tokenized_text: tokenized text list
    :param category: category param
    :param doc_id_number: document의 id number (index)
    :param word2topic_dict: 함수 word2topic의 dictionary 반환값
    :return: 2차원 ndarray [[education_1, topic1, topic2, ..],[sports_2, topic1, topic1,..],..]
    """
    doc_to_topic_list = [category + '_' + str(doc_id_number)]
    for word in tokenized_text:
        if word in word2topic_dict.keys():
            doc_to_topic_list.append(word2topic_dict[word])

    return doc_to_topic_list


class LabeledLineSentenceTraining(object):
    def __init__(self, word2topic_dict):
        self.labels_list = word2topic_dict

    def __iter__(self):
        for idx, doc in enumerate(DATA_FRAME['joined_sentence']):
            words_doc = doc.split(' ')
            category = DATA_FRAME['category'][idx]
            tags_doc = map_doc_to_topic(words_doc, category, idx, word2topic_dict)
            yield models.doc2vec.LabeledSentence(words=words_doc, tags=tags_doc)

    def to_array(self):
        self.sentences = []

        for idx, doc in enumerate(DATA_FRAME['joined_sentence']):
            words_doc = doc.split(' ')
            category = DATA_FRAME['category'][idx]
            tags_doc = map_doc_to_topic(words_doc, category, idx, word2topic_dict)
            self.sentences.append(models.doc2vec.LabeledSentence(words=words_doc, tags=tags_doc))

        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


def save_model(directory):
    cur_path = os.getcwd()

    if not os.path.exists(directory):
        os.makedirs(directory)

    fname = cur_path + '/' + directory + '/' + 'n_docs' + str(N_DOCUMENTS) + 'n_topics' + str(N_TOPICS) + '.model'
    model.save(fname)
    print('save!!')

def main():
    word2topic_dict = word2topic()

    it = LabeledLineSentenceTraining(word2topic_dict)

    model = models.Doc2Vec(size=100, window=10, min_count=4, dm=1, dbow_words=1, workers=50, alpha=0.025,
                           min_alpha=0.025)
    model.build_vocab(it.to_array())

    for epoch in range(10):
        print('start')
        start = time()

        model.train(it.sentences_perm(), total_examples=model.corpus_count, epochs=model.iter)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay

        stop = time()
        duration = stop - start
        print('epoch:', epoch, ' duration: ', duration)


if __name__=="__main__":

    main()

    save_model('model')


