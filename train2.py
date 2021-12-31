import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import json
import pickle
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

lemmatizer = WordNetLemmatizer()  #object of WordNetLemmatizer class

words=[]
classes = []
documents = []
ignore_words = ['?', '!'] 
intent_json = json.load(open('E:\Python projects\chatbot\intents.json','r',encoding='utf-8'))


for Int in intent_json['intents']:
    for pattern in Int['patterns']:

        #tokenize each word - split words into an array
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, Int['tag']))

        # add to our classes list
        if Int['tag'] not in classes:
            classes.append(Int['tag'])

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(each_word.lower()) for each_word in words if each_word not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print ('\n\n\n',len(documents), "documents")
# classes = intents
print ('\n\n\n',len(classes), "classes", classes)
# words = all words, vocabulary
print ('\n\n\n',len(words), "unique lemmatized words", words,'\n\n\n')


pickle.dump(words,open('E:\Python projects\chatbot\words.pkl','wb'))                          #serializing word object
pickle.dump(classes,open('E:\Python projects\chatbot\classes.pkl','wb'))                      #serializing classes object

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
    #print('\n\noutput_row before indexing :',output_row)
    output_row[classes.index(doc[1])] = 1
    #print('\n\noutput_row after indexing :',output_row)

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training,dtype=object)
print(training.shape)
# create train and test lists. X - patterns, Y - intents
x_train = list(training[:,0])
y_train = list(training[:,1])

print("Training data created")

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='sigmoid'))

x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])



#model.fit(train_data, epochs=10, validation_data=valid_data)

#fitting and saving the model 
hist = model.fit(x_train,y_train, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")
