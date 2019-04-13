import codecs
import thulac
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import sklearn.preprocessing
import keras
import pandas as pd
import gensim

#读取数据
def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open(filename,'r',encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(content)
                    labels.append(label)
            except:
                pass
    return contents, labels


#将数据进行分词处理
def cut_text(texts,filename):
    print('cut text...')
    count = 0
    cut = thulac.thulac(seg_only=True)
    train_text = []
    for text in texts:
        count += 1
        if count % 2000 == 0:
            print(count)
        train_text.append(cut.cut(text, text=True)) #分词结果以空格间隔，每个fact一个字符串
    print(len(train_text))

    fileObject = codecs.open(filename, "w", "utf-8")  #必须指定utf-8否则word2vec报错
    for ip in train_text:
        fileObject.write(ip)
        fileObject.write('\n')
    fileObject.close()
    print('cut text over')
    return train_text


#用分词后的文本文档训练word2vec词向量模型并保存，这里使用了默认的size=100，即每个词由100维向量表示。
def word2vec_train():
    print("start generate word2vec model...")
    sentences = gensim.models.word2vec.Text8Corpus("./data/trained.txt")
    model = gensim.models.word2vec.Word2Vec(sentences)         #默认size=100 ,100维
    model.save('./data/word2vec')
    print('finished and saved!')
    return model


#将文本转化为数字序列
def data_processing(filepath):
    train_data = []
    with open(filepath,encoding='utf-8', errors='ignore') as f:
        train_data = f.read().splitlines()
    maxlen = 150
    # 词袋模型的最大特征束
    max_features = 2000

    # 设置分词最大个数 即词袋的单词个数
    # with open('./predictor/model/tokenizer.pickle', 'rb') as f:
    #   tokenizer = pickle.load(f)
    tokenizer = Tokenizer(num_words=max_features, lower=True)  # 建立一个max_features个词的字典
    tokenizer.fit_on_texts(train_data)  # 使用一系列文档来生成token词典，参数为list类，每个元素为一个文档。可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小。
    #global word_index
    #word_index = tokenizer.word_index  # 长度为508242
    # with open('./predictor/model/tokenizer_large.pickle', 'wb') as handle:
    #   pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print("tokenizer has been saved.")
    # self.tokenizer.fit_on_texts(train_data)  # 使用一系列文档来生成token词典，参数为list类，每个元素为一个文档。可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小。

    sequences = tokenizer.texts_to_sequences(train_data)
#    sequences=tokenizer.sequences_to_matrix(sequences)
    return sequences


#获取数据与标签
train_filename = './data/train.txt'
test_filename = './data/test.txt'
x_train1,y_train1= read_file(train_filename)
x_test1,y_test1= read_file(test_filename)


#将标签变成to_categorical格式
train_y = pd.Series(y_train1)
test_y = pd.Series(y_test1)
y_labels = list(train_y.value_counts().index)
le = sklearn.preprocessing.LabelEncoder()
le.fit(y_labels)
num_labels = len(y_labels)
y_train = keras.utils.to_categorical(train_y.map(lambda x: le.transform([x])[0]), num_labels)
y_test = keras.utils.to_categorical(test_y.map(lambda x: le.transform([x])[0]), num_labels)


#对data进行处理，并设置每一句的最大长度为150
train_path = './data/trained.txt'
test_path = './data/tested.txt'
# x_train2=cut_text(x_train1,train_path)
# y_train2=cut_text(x_test1,test_path)
train_x_seq = data_processing(train_path)
test_x_seq = data_processing(test_path)
x_train = sequence.pad_sequences(train_x_seq, maxlen=150)
x_test = sequence.pad_sequences(test_x_seq, maxlen=150)








