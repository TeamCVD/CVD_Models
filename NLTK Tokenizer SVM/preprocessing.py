import numpy as np
import tensorflow as tf
from nltk.tokenize import RegexpTokenizer

def tokenize_data(data):
    print("--------------------Tokenizing data--------------------")
    tokenizer = RegexpTokenizer(r'\w+')
    List_Tokenized_Words = [tokenizer.tokenize(x) for x in data['code']]

    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=False)
    tokenizer.fit_on_texts(List_Tokenized_Words)

    list_tokenized = tokenizer.texts_to_sequences(List_Tokenized_Words)

    x_train = tf.keras.preprocessing.sequence.pad_sequences(list_tokenized,
                                                             maxlen=512,
                                                             padding='post')
    x_train = x_train.astype(np.int64)
    
    return x_train

