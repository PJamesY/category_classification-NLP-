from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from make_cnn_dataset import MakeCnnDataset
import numpy as np


class CNN:

    def __init__(self, embedding_dim=100, filter_size=[3, 4, 5], num_filters=512,
                 drop_rate=0.5, epochs=100, batch_size=30, pre_trained_embedding_layer=False):
        """

        :param embedding_dim: embedding dimension
        :param filter_size: filter size list
        :param num_filters: number of filters
        :param drop_rate: drop rate
        :param pre_trained_embedding_layer: whether use embedding_layer or not (boolean)
        """
        self.embedding_dim = embedding_dim  # kor2vec embedding size가 100이므로 100으로 맞춰주었다.
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.drop_rate = drop_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.pre_trained_embedding_layer = pre_trained_embedding_layer

        print("Loading Data ...")
        cnn_dataset = MakeCnnDataset("Komoran")
        self.X, self.y, self.word_to_index, self.vocabulary = cnn_dataset.make_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=0.2, random_state=42)
        # self.X.shape -> (number of sentences, length of sentence)
        # self.y.shape -> (number of sentences, number of category)

        print("Setting parameter...")
        self.sentence_length = self.X.shape[1]   # length of sentence
        self.voca_size = len(self.vocabulary)  # number of words
        self.channel = 1  # number of channels : embding하는 갯수(Word2vec, glovec)
        self.classes = 7  # number of class

    def model_construction_train(self):
        print('creating model..')

        # input layer
        inputs = Input(shape=(self.sentence_length,), dtype='int32')

        # embedding layer
        if self.pre_trained_embedding_layer:
            # load the whole embedding into memory
            embeddings_index = dict()
            f = open('vectors_on_batch.txt')
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[2:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()
            print('Loaded %s word vectors.' % len(embeddings_index))

            # create a weight matrix for words in training docs
            embedding_matrix = np.zeros((self.voca_size, self.embedding_dim))
            for word, i in self.word_to_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

            embedding = Embedding(self.voca_size,
                                  self.embedding_dim,
                                  weights=[embedding_matrix],
                                  input_length=self.sentence_length,
                                  trainable=False)
        else:
            embedding = Embedding(input_dim=self.voca_size,
                                  output_dim=self.embedding_dim,
                                  input_length=self.sentence_length)(inputs)

        # Reshape layer
        reshape = Reshape((self.sentence_length, self.embedding_dim, self.channel))(embedding)

        # Convolution layer
        conv_0 = Conv2D(self.num_filters, kernel_size=(self.filter_size[0], self.embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape)
        conv_1 = Conv2D(self.num_filters, kernel_size=(self.filter_size[1], self.embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape)
        conv_2 = Conv2D(self.num_filters, kernel_size=(self.filter_size[2], self.embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape)

        # MAX Pooling
        maxpool_0 = MaxPool2D(pool_size=(self.sentence_length - self.filter_size[0] + 1, 1),
                              strides=(1, 1), padding='valid')(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(self.sentence_length - self.filter_size[1] + 1, 1),
                              strides=(1, 1), padding='valid')(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(self.sentence_length - self.filter_size[2] + 1, 1),
                              strides=(1, 1), padding='valid')(conv_2)

        # Concatenate, Flatten, Dense
        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(self.drop_rate)(flatten)
        output = Dense(units=self.classes, activation='softmax')(dropout)

        # Create model
        model = Model(inputs=inputs, outputs=output)

        checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1,
                                     save_best_only=True, mode='auto')
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

        print('Training_model ....')
        model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs,
                  verbose=1, callbacks=[checkpoint], validation_data=(self.X_test, self.y_test))

        return model

    def train_save_model(self):
        model = self.model_construction_train()

        model_json = model.to_json()

        with open("model.json", "w") as json_file:
            json_file.write(model_json)


if __name__ == "__main__":
    cnn_model = CNN()
    cnn_model.train_save_model()