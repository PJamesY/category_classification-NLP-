import os
from collections import Counter
from time import time
import random
import numpy as np
import pandas as pd
import math
import copy
from keras.layers import Dense, Dot, Embedding, Input, Reshape
from keras.models import Model
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
from konlpy.tag import Okt, Komoran
from preprocessing import ArticlePreProcessing
np.random.seed(777)
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

df = pd.read_csv('data/article.csv')


class Kor2Vec:

    def __init__(self, konlpy="Okt", min_count=5, sampling_rate=0.0001,
                 window_size=4, embedding_dim=100, epochs=100001, batch_size=512):
        self.konlpy = konlpy
        self.min_count = min_count
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size

        self.df = df.astype(str)
        self.df = pd.concat([pd.DataFrame({'article': doc, 'category': row['category']}, index=[0])
                            for _, row in self.df.iterrows()
                            for doc in row['article'].split('.') if doc != ''])

        preprocessing = ArticlePreProcessing(self.df, remove=False)
        self.df = preprocessing.text_pre_processing('article')

    '''
        Step 1 : Pre-process Data.
    '''

    def build_dataset(self, column='article'):

        if self.konlpy == "Okt":
            konlpy = Okt()
        else:
            konlpy = Komoran()

        words = list(self.df[column].apply(lambda sentence: sentence.strip().split()))
        # words.shape = (문장의 갯수, 단어의 갯수)

        # 단어의 갯수를 세고, 지정한 min_count보다 높거나, UNK라고 되어 있는것만 word_counter list 에 넣는다
        word_counter = [['UNK', -1]]
        word_counter.extend(Counter([word for sentence in words for word in sentence]).most_common())
        word_counter = [item for item in word_counter if item[1] >= self.min_count or item[0] == "UNK"]

        # 단어와 index쌍으로 이루어진 dictionary를 만든다
        word2index = dict()
        for word, _ in word_counter:
            word2index[word] = len(word2index) + 1
        index2word = dict(zip(word2index.values(), word2index.keys()))

        # word를 형태소 단위로 쪼개고 word와 tag로 이루어진 dictionary를 만든다
        word_to_tag_dict = dict()
        tag_list = list()

        for word in word2index:
            word_tag_list = list()
            for tag in konlpy.pos(word, norm=True):
                word_tag_list.append(tag)

            word_to_tag_dict[word2index[word]] = word_tag_list
            tag_list += word_tag_list

        # tag(형태소)의 갯수를 센다
        tag_counter = Counter(tag_list).most_common()

        # 형태소와 index쌍으로 이루어진 dictionary를 만든다
        tag2index = dict()
        for tag, _ in tag_counter:
            tag2index[tag] = len(tag2index) + 1
        index2tag = dict(zip(tag2index.values(), tag2index.keys()))

        # word index와 word에 대한 형태소 index 모음 쌍으로 dictionary를 만든다
        word_idx_to_tag_idx = dict()

        for word_index, tag_list in word_to_tag_dict.items():
            tag_index_list = list()
            for tag in tag_list:
                tag_index_list.append(tag2index[tag])
            word_idx_to_tag_idx[word_index] = tag_index_list

        # 단어로 이루어진 코퍼스를 만든다
        word_corpus = list()
        unk_count = 0
        for sentence in words:
            s = list()
            for word in sentence:
                if word in word2index:
                    index = word2index[word]
                else:
                    index = word2index['UNK']
                    unk_count += 1
                s.append(index)
            word_corpus.append(s)
        word_counter[0][1] = max(1, unk_count)

        word_corpus = self.sub_sampling(word_corpus, word_counter, word2index)

        # tag로 이루어진 코퍼스를 만든다
        tag_corpus = copy.deepcopy(word_corpus)
        for idx1, sentence in enumerate(word_corpus):
            for idx2, word in enumerate(sentence):
                tag_corpus[idx1][idx2] = word_idx_to_tag_idx[word]

        return word_corpus, tag_corpus, word2index, word_idx_to_tag_idx, tag2index

    def sub_sampling(self, word_corpus, word_counter, word2index):
        total_words = sum([len(sentence) for sentence in word_corpus])
        probability_dict = dict()
        for word, count in word_counter:
            frequency = count / total_words
            probability = max(0, 1 - math.sqrt(self.sampling_rate / frequency))
            probability_dict[word2index[word]] = probability

        new_word_corpus = list()
        for sentence in word_corpus:
            s = list()
            for word in sentence:
                probability = probability_dict[word]
                if random.random() > probability:
                    s.append(word)
            new_word_corpus.append(s)

        return new_word_corpus

    def generating_wordpairs(self, indexed_corpus, tag_size):
        X = []
        Y = []
        for row in indexed_corpus:
            x, y = skipgrams(sequence=row, vocabulary_size=tag_size, window_size=self.window_size,
                             negative_samples=1.0, shuffle=True, categorical=False, sampling_table=None, seed=None)

            idx_list = [idx for idx, ls in enumerate(x) if type(x[idx][1]) == int]

            x = [x[idx] for idx, _ in enumerate(x) if idx not in idx_list]
            y = [y[idx] for idx, _ in enumerate(y) if idx not in idx_list]

            X = X + list(x)
            Y = Y + list(y)
        return X, Y

    '''
        Step 2 : kor2vec Keras Model
    '''

    def construction_model(self, tag_size):
        input_target = Input((3,))
        input_context = Input((3,))

        embedding_layer = Embedding(tag_size, self.embedding_dim, input_length=3)

        target_embedding = embedding_layer(input_target)
        target_embedding = Reshape((3, self.embedding_dim))(target_embedding)
        target_embedding_flatten = Reshape((self.embedding_dim * 3,))(target_embedding)
        target_dense = Dense(self.embedding_dim, activation='relu')(target_embedding_flatten)
        target_embedding_final = Reshape((self.embedding_dim, 1))(target_dense)

        context_embedding = embedding_layer(input_context)
        context_embedding = Reshape((3, self.embedding_dim))(context_embedding)
        context_embedding_flatten = Reshape((self.embedding_dim * 3,))(context_embedding)
        context_dense = Dense(self.embedding_dim, activation='relu')(context_embedding_flatten)
        context_embedding_final = Reshape((self.embedding_dim, 1))(context_dense)

        hidden_layer = Dot(axes=1)([target_embedding_final, context_embedding_final])
        hidden_layer = Reshape((1,))(hidden_layer)

        output = Dense(16, activation='sigmoid')(hidden_layer)
        output = Dense(1, activation='sigmoid')(output)

        model = Model(inputs=[input_target, input_context], outputs=output)
        # model.summary()
        model.compile(loss='binary_crossentropy', optimizer='sgd')
        return model

    def training_model(self, model, indexed_corpus, tag_size):
        for i in range(self.epochs):
            idx_batch = np.random.choice(len(indexed_corpus), self.batch_size)
            x, y = self.generating_wordpairs(np.array(indexed_corpus)[idx_batch].tolist(), tag_size)

            word_target, word_context = zip(*x)
            word_target = np.array(word_target, dtype=np.int32)
            word_context = np.array(word_context, dtype=np.int32)

            target = list()
            context = list()
            label = np.zeros((1,))
            idx = np.random.randint(0, len(y)-1)

            target.append(word_target[idx][0])
            target.append(word_target[idx][1])
            target.append(word_target[idx][2])
            target = np.array(target)
            target = target.reshape(1, 3)

            context.append(word_context[idx][0])
            context.append(word_context[idx][1])
            context.append(word_context[idx][2])
            context = np.array(context)
            context = context.reshape(1, 3)

            label[0, ] = y[idx]
            loss = model.train_on_batch([target, context], label)

            if i % 1000 == 0:
                print("Iteration {}, loss={}".format(i, loss))

        return model

    def save_vectors(self, file_path, tag_size, model, tag2index):
        f = open(file_path, 'w')
        f.write('{} {}\n'.format(tag_size-1, self.embedding_dim))
        vectors = model.get_weights()[0]
        for word, i in tag2index.items():
            f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i-1, :])))))
        f.close()
        return file_path

    def main(self):
        time_start = time()
        time_check = time()

        word_corpus, tag_corpus, word2index, word_idx_to_tag_idx, tag2index = self.build_dataset()

        # check voca_size, tag_size
        voca_size = len(word2index)
        tag_size = len(tag2index)
        print('number of voca :', voca_size)
        print('number of tags :', tag_size)
        print('word, tag was indexed in \t{time} sec'.format(time=time() - time_check))
        time_check = time()

        # make corpus
        corpus = copy.deepcopy(tag_corpus)
        for idx, sentence in enumerate(tag_corpus):
            corpus[idx] = sequence.pad_sequences(sentence, maxlen=3).tolist()
        print("Corpus was loaded in\t{time} sec".format(time=time() - time_check))
        time_check = time()

        # make model
        model = self.construction_model(tag_size)
        print("Model was constructed in\t{time} sec".format(time=time() - time_check))
        time_check = time()

        # train model
        model = self.training_model(model, corpus, tag_size)
        print("Traning was done in\t{time} sec".format(time=time() - time_check))
        time_check = time()

        # save vector
        save_path = self.save_vectors('vectors_on_batch.txt', tag_size, model, tag2index)
        print("Trained vector was saved in\t{time} sec".format(time=time() - time_check))

        print("Done: overall process consumes\t{time} sec".format(time=time() - time_start))


if __name__ == "__main__":
    kor2vec = Kor2Vec("Okt")
    kor2vec.main()
