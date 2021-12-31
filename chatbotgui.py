import re
#import long_responses as long
from textblob import TextBlob    #for calculating the polarity of the input
import speech_recognition as SRG  #for taking voice input
import pyttsx3

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
    return re

from tkinter import *
def send():
    msg = messageWindow.get("1.0",'end-1c').strip()
    messageWindow.delete("0.0",END)

    if msg != '':
        chatWindow.config(state=NORMAL)
        chatWindow.insert(END, "You: " + msg + '\n\n')
        chatWindow.config(foreground="#FFFFFF", font=("Verdana", 12 ))
    
        res = chatbot_response(msg)
        chatWindow.insert(END, "Bot: " + res + '\n\n')
            
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
messageWindow = Text(root,bg='white',width=30,height=2,bd=4,font=('calibiri',12))

messageWindow.place(x=6 ,y=400,height=60 ,width=310)

#creating and placing submit button
Button= Button(root ,text='Send',bg='light blue',width=12,height=5,font=('Arial',14), command = send)

Button.place(x=330,y=400,height=60,width=100)

#creating and placing scrollbar
scrollbar=Scrollbar(root,command=chatWindow.yview())

scrollbar.place(x=422,y=5,height=385)

#running the main loop for displaying the window for chatbot
root.mainloop()