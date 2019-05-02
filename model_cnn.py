from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
import make_cnn_dataset

print("Loading Data ...")
X, y, word_to_index, vocabulary = make_cnn_dataset.make_data()

# X.shape -> (159119, 252)
# y.shape -> (159119, 8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Setting parameter ...')
# model parameter
SEQUENCE_LENGTH = X.shape[1] # 252(문장 길이)
VOCABULARY_SIZE = len(vocabulary) # 50972(단어 갯수)
EMBEDDING_DIM = 256
FILTER_SIZES = [3,4,5]
NUM_FILTERS = 512
DROP_RATE = 0.5
CHANNEL = 1 # 채널 갯수 : embding하는 갯수(Word2vec, glovec)
CLASSES = 8

# Learning parameter
EPOCHS = 100
BATCH_SIZE = 30
PRE_TRAINED_EMBEDDING_LAYER = False


print('creating Model  ...')
# input layer
inputs = Input(shape=(SEQUENCE_LENGTH,), dtype='int32')
# embedding layer
if PRE_TRAINED_EMBEDDING_LAYER:
    pass
else:
    embedding = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBEDDING_DIM, input_length=SEQUENCE_LENGTH)(inputs)
reshape = Reshape((SEQUENCE_LENGTH, EMBEDDING_DIM, CHANNEL))(embedding)

# Convolution layer
conv_0 = Conv2D(NUM_FILTERS, kernel_size=(FILTER_SIZES[0], EMBEDDING_DIM), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(NUM_FILTERS, kernel_size=(FILTER_SIZES[1], EMBEDDING_DIM), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(NUM_FILTERS, kernel_size=(FILTER_SIZES[2], EMBEDDING_DIM), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

# MAX Pooling
maxpool_0 = MaxPool2D(pool_size=(SEQUENCE_LENGTH - FILTER_SIZES[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(SEQUENCE_LENGTH - FILTER_SIZES[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(SEQUENCE_LENGTH - FILTER_SIZES[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

# Concatenate, Flatten, Dense
concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(DROP_RATE)(flatten)
output = Dense(units=CLASSES, activation='softmax')(dropout)

# Create model
model = Model(inputs=inputs, outputs=output)

checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
print('Training_model ....')
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[checkpoint], validation_data=(X_test, y_test))


print(1)