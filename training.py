'''
import json 

import numpy as np 

import tensorflow as tf 

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder

"""

    define few simple intents and bunch of messages 

    that corresponds to those intents and

    also map some responses according to each intent category.
    

    create a JSON file named “intents.json”


"""


data = json.load(open('intents.json','r',encoding="utf8"))
    

training_sentences = []

training_labels = []

labels = []

responses = []


"""

The variable “training_sentences” holds all the training data (which are the sample messages in each intent category) 

and the “training_labels” variable holds all the target labels correspond to each training data.

Then we use “LabelEncoder()” function provided by scikit-learn to convert the target labels into a model understandable form.

"""


for intent in data['intents']:

    for pattern in intent['patterns']:
        training_sentences.append(pattern)

        training_labels.append(intent['tag'])

    responses.append(intent['responses'])
    

    if intent['tag'] not in labels:

        labels.append(intent['tag'])
        
num_classes = len(labels)



lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)


"""

Next, we vectorize our text data corpus by using the “Tokenizer” class and it allows us to limit our vocabulary size

up to some defined number. When we use this class for the text pre-processing task,

by default all punctuations will be removed, turning the texts into space-separated sequences of words,

and these sequences are then split into lists of tokens.

They will then be indexed or vectorized. We can also add “oov_token” which is a value for

“out of token” to deal with out of vocabulary words(tokens) at inference time.

"""

vocab_size = 1000

embedding_dim = 16

max_len = 20

oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)


#The “pad_sequences” method is used to make all the training text sequences into the same size.


#model training

model = Sequential()  #sequential model class of keras

model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))

model.add(GlobalAveragePooling1D())

model.add(Dense(16, activation='relu'))

model.add(Dense(16, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy', 

              optimizer='adam', metrics=['accuracy'])


#model.summary()


#training the method by calling the fit method

#epochs = 500

#history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)


model.save("chat_model")

import pickle


# to save the fitted tokenizer

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(Tokenizer(num_words=vocab_size, oov_token=oov_token), handle, protocol=pickle.HIGHEST_PROTOCOL)
    

# to save the fitted label encoder

with open('label_encoder.pickle', 'wb') as ecn_file:

    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
'''
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import tensorflow as tf
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!'] 
intents =json.load(open('E:\Python projects\chatbot\intents.json','r',encoding='utf8'))


for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('E:\Python projects\chatbot\words.pkl','wb'))
pickle.dump(classes,open('E:\Python projects\chatbot\classes.pkl','wb'))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training,dtype=object)
# create train and test lists. X - patterns, Y - intents
x_train = list(training[:,0])
y_train = list(training[:,0])

#test_x = list(training[:,0:])
#test_y = list(training[:,1:])
print("Training data created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='sigmoid'))

#train_x=tf.ragged.constant(test_x)  #non rectangular python sequence to tensor conversion
#train_y=tf.ragged.constant(train_y) #non rectangular python sequence to tensor conversion
#test_x=tf.ragged.constant(test_x)   #non rectangular python sequence to tensor conversion
#test_y=tf.ragged.constant(test_y)   #non rectangular python sequence to tensor conversion

#train_data = tf.data.Dataset.from_tensor_slices(train_x, train_y)
#print(train_data)
#valid_data = tf.data.Dataset.from_tensor_slices(test_x, test_y)
#print(valid_data)

#ragged_tensor.RaggedTensor.__truediv__ = math_ops.truediv
#ragged_tensor.RaggedTensor.__rtruediv__ = _right(math_ops.truediv)


# Dummy methods
#def _dummy_bool(_):
"""Dummy method to prevent a RaggedTensor from being used as a Python bool."""
 # raise TypeError("RaggedTensor may not be used as a boolean.")

x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)

#ragged_tensor.RaggedTensor.__bool__ = _dummy_bool
#ragged_tensor.RaggedTensor.__nonzero__ = _dummy_bool

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(x_train,y_train, epochs=200, batch_size=5, verbose=1)
#hist = model.fit(train_x, train_y, epochs=200,validation_data=valid_data,verbose=1,batch_size=5)
model.save('chatbot_model.h5', hist)

print("model created")
