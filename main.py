import re
#import long_responses as long
from textblob import TextBlob    #for calculating the polarity of the input
import speech_recognition as SRG  #for taking voice input
import pyttsx3
import time
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
#import text2emotion as te


from keras.models import load_model
model = load_model('E:\Python projects\chatbot_model.h5')
import json
import random
intents = json.load(open('E:\Python projects\chatbot\intents.json','r',encoding='utf8'))
words = pickle.load(open('E:\Python projects\chatbot\words.pkl','rb'))
classes = pickle.load(open('E:\Python projects\chatbot\classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def sentiment(msg):
    #print(msg)
    if isinstance(msg, list):
        msg=' '.join([str(x) for x in msg])
    #print(msg)
    text=TextBlob(msg)
    #print(text.sentiment.polarity)
    return text.sentiment.polarity

def speechtext(voice_input):
    engine=pyttsx3.init()
    engine.say(voice_input)
    engine.runAndWait()


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result


def get_response(user_input):
    user_input=user_input.lower()
    split_message = re.split(r'\s+|[,;?!.-]\s*',user_input) #regex expression to remove all the special characters
    response = sentiment(split_message)
    #feature=te.get_emotion(split_message)
    print(split_message)
    if response<0:
        return 'Negative'
    elif response>0 and response<=1:
        return 'Positive'
    else:
        return 'Neutral'

def chatbot_response(msg):
    ints = predict_class(msg, model)
    print(ints)
    print("***\n",intents)
    res = getResponse(ints, intents)
    return res




from tkinter import *
def send():
    msg = messageWindow.get("1.0",'end-1c').strip()
    messageWindow.delete("0.0",END)

    if msg != '':
        chatWindow.config(state=NORMAL)
        chatWindow.insert(END, "You: " + msg + '\n\n')
        chatWindow.config(foreground="#FFFFFF", font=("Verdana", 12 ))
    
        res = get_response(msg)
        chatres=chatbot_response(msg)
        chatWindow.insert(END, "Bot: " + chatres +' \n\n')
        speechtext(res)
        #chatWindow.insert(END, "Bot: " + speechtext(res) +' \n\n')
  
        chatWindow.config(state=DISABLED)
        chatWindow.yview(END)

root=Tk()

#title of the application 
root.title('Friendly bot')

#setting window dimensions
root.geometry('450x500')

#setting the resizable property to false
root.resizable(width=False,height=False)

main_menu=Menu(root)

file_menu=Menu(root)

file_menu.add_command(label='New')

file_menu.add_command(label='Save')

file_menu.add_command(label='Save as..')

file_menu.add_command(label='Quit')

main_menu.add_cascade(label='File',menu=file_menu)

main_menu.add_command(label='Edit')

main_menu.add_command(label='Quit')

root.config(menu=main_menu)

#creating and placing chat window
chatWindow= Text(root,bd=1,bg='black',width=50,height=10)

chatWindow.place(x=8,y=8,height=385,width=420)

#creating and placing user msg window
messageWindow = Text(root,bg='white',width=30,height=2,bd=2,font=('calibiri',12))

messageWindow.place(x=120 ,y=400,height=60 ,width=310)

#creating and placing submit button
Button= Button(root ,text='Send',bg='light blue',width=12,height=5,font=('Arial',14), command = send)

Button.place(x=6,y=400,height=60,width=100)

#creating and placing scrollbar
scrollbar=Scrollbar(root,command=chatWindow.yview())

scrollbar.place(x=422,y=5,height=385)

#running the main loop for displaying the window for chatbot
root.mainloop()

'''def message_probability(user_message, recognised_words, single_response=False, required_words=[]):
    message_certainty = 0
    has_required_words = True

    # Counts how many words are present in each predefined message
    for word in user_message:
        if word in recognised_words:
            message_certainty += 1

    # Calculates the percent of recognised words in a user message
    percentage = float(message_certainty) / float(len(recognised_words))

    # Checks that the required words are in the string
    for word in required_words:
        if word not in user_message:
            has_required_words = False
            break

    # Must either have the required words, or be a single response
    if has_required_words or single_response:
        return int(percentage * 100)
    else:
        return 0
'''


'''def check_all_messages(message):
    highest_prob_list = {}

    # Simplifies response creation / adds it to the dict
    #def response(bot_response, list_of_words, single_response=False, required_words=[]):
    def response(bot_response):
        nonlocal highest_prob_list
        highest_prob_list[bot_response] =sentiment(message)
    response('hello')
    print(highest_prob_list)
    # Responses -------------------------------------------------------------------------------------------------------
    response('Hello!', ['hello', 'hi', 'hey', 'sup', 'heyo','heya','hola'], single_response=True)
    response('See you!', ['bye', 'goodbye' ,'see you'], single_response=True)
    response('I\'m doing fine, and you?', ['how', 'are', 'you', 'doing'], required_words=['how'])
    response('Okay. That\'s Great!!!',['fine'],required_words=['fine'])
    response('You\'re welcome!', ['thank', 'thanks'], single_response=True)
    response('Thank you! It was nice talking with you as well.', ['nice', 'talking', 'with ', 'you', 'great'], required_words=['talking'])
    response('Good Night to you as well..',['good' ,'night'],required_words=['good' ,'night'],single_response=True)
        # Longer responses
    response(long.R_ADVICE, ['give', 'advice'], required_words=['can','advice'])
    response(long.R_EATING, ['what', 'you', 'eat'], required_words=['you', 'eat'])

    best_match = max(highest_prob_list, key=highest_prob_list.get)
    #print(highest_prob_list)
    #print(f'Best match = {best_match} | Score: {highest_prob_list[best_match]}')

    return long.unknown() if highest_prob_list[best_match] < 1 else best_match
'''



# Testing the response system
'''choice=input('How do you want to communicate with me ? text or speech : ')
if choice =='text':
    print('Okay cool😁\n...')
    while True:
        print('Bot: ' + get_response(input('You: ')))
else:
    print('Sure why not😃...') 
    while True:
        inp=speech()
        speechtext(inp)
        print('Bot: ' +inp)
'''
    