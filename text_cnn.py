from data_processing import x_train,x_test,y_train,y_test
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, Concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
from numpy import *
import keras
import gensim


#使用预训练的word2vec文件进行cnn_text
def cnn_w2v(max_features=2000,embedding_dims=100,filters=250,maxlen=150):
    # CNN参数
    kernel_size = 3
    model = gensim.models.Word2Vec.load('./data/word2vec')
    word2idx = {"_PAD": 0}  # 初始化 `[word : token]` 字典，后期 tokenize 语料库就是用该词典。
    vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
    # 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
    embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
    print('Found %s word vectors.' % len(model.wv.vocab.items()))
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i + 1
        embeddings_matrix[i + 1] = vocab_list[i][1]

    model = keras.Sequential()
    # 使用Embedding层将每个词编码转换为词向量
    model.add(Embedding(len(embeddings_matrix),       #表示文本数据中词汇的取值可能数,从语料库之中保留多少个单词。 因为Keras需要预留一个全零层， 所以+1
                                embedding_dims,       # 嵌入单词的向量空间的大小。它为每个单词定义了这个层的输出向量的大小
                                weights=[embeddings_matrix], #构建一个[num_words, EMBEDDING_DIM]的矩阵,然后遍历word_index，将word在W2V模型之中对应vector复制过来。换个方式说：embedding_matrix 是原始W2V的子集，排列顺序按照Tokenizer在fit之后的词顺序。作为权重喂给Embedding Layer
                                input_length=maxlen,     # 输入序列的长度，也就是一次输入带有的词汇个数
                                trainable=False        # 我们设置 trainable = False，代表词向量不作为参数进行更新
                        ))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # 池化
    model.add(keras.layers.GlobalMaxPooling1D())

    model.add(Dense(10, activation='softmax')) #第一个参数units: 全连接层输出的维度，即下一层神经元的个数。
    model.add(Dropout(0.2))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    return model


def text_cnn(maxlen=150,max_features=2000,embedding_dims=100,filters = 250):
    #Inputs
    seq = Input(shape=[maxlen],name='x_seq')

    #Embedding layers
    emb = Embedding(max_features,embedding_dims)(seq)

    # conv layers
    convs = []
    filter_sizes = [2,3,4,5]
    for fsz in filter_sizes:
        conv1 = Conv1D(filters,kernel_size=fsz,activation='tanh')(emb)
        pool1 = MaxPooling1D(maxlen-fsz+1)(conv1)
        pool1 = Flatten()(pool1)
        convs.append(pool1)
    merge = Concatenate(axis=1)(convs)

    out = Dropout(0.5)(merge)
    output = Dense(32,activation='relu')(out)

    output = Dense(10,activation='sigmoid')(output)

    model = Model([seq],output)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


if __name__=="__main__":
    model = text_cnn()
    #model=cnn_w2v
    batch_size = 128
    epochs = 10
    model.fit(x_train, y_train,
              validation_split=0.1,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True)
    scores = model.evaluate(x_test, y_test)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))





