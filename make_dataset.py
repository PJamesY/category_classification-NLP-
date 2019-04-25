import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# dataframe 가져오기(load)
df = pd.read_pickle('data/morphs_article.pkl')

# array 형태로 string sentence - > ['sentence1','sentence2','...']
docs = np.array(df['modified_article_sentence'])

# Constant
# 한 문장에 사용할 단어 갯수

def sentence_length(df):
    max_length = 0
    for word_tag_ls in df['article_modified_pos']:
        if len(word_tag_ls) >= max_length:
            max_length = len(word_tag_ls)

    return max_length

MAX_SEQ_LENGTH = sentence_length(df) # 최대 문장 길이
MAX_NUM_WORDS = 15000




def make_data():


    # tokenizer
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(docs)
    sequences = tokenizer.texts_to_sequences(docs)

    word_to_index = tokenizer.word_index
    vocabulary = word_to_index.keys()

    sentence = [len(sentence.split()) for sentence in docs]

    print('Text informations:')
    print('sentence max length: %i / min length: %i / mean length: %i / limit length: %i' % (np.max(sentence),
                                                                                    np.min(sentence),
                                                                                    np.mean(sentence),
                                                                                    MAX_SEQ_LENGTH))
    print('총 단어 갯수: %i / 한 문장 속에 제한: %i' % (len(word_to_index), MAX_NUM_WORDS))


    # make X data set
    # Padding all sequences to same length of `MAX_SEQ_LENGTH`, 최대 문장 길이보다 작은 문장은 나머지 단어들은 padding 0으로 처리 된다
    X = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, padding='post')

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
                        'it': 4,
                        'sports': 5,
                        'entertainment': 6,
                        'travel': 7
                        }

    # dataframe에 카테고리들을 int label 컬럼을 추가시킨다
    df['category2label'] = df['category'].apply(lambda category: cateory_to_label[category])

    y = to_categorical(list(df['category2label']))

    # y
    # array([[0., 0., 0., ..., 0., 1., 0.],
    #        [1., 0., 0., ..., 0., 0., 0.],
    #        [0., 1., 0., ..., 0., 0., 0.],
    #        ...,
    #        [0., 0., 1., ..., 0., 0., 0.],
    #        [1., 0., 0., ..., 0., 0., 0.],
    #        [0., 0., 1., ..., 0., 0., 0.]], dtype=float32)
    return X, y, word_to_index, vocabulary

# if __name__ == "__main__":
#     X,y,word_to_index, voca = make_data()
#     print(X.shape)




