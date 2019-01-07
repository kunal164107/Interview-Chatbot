from tkinter import *
from tkinter import scrolledtext
from test_model import chatbot
from tkinter import LEFT,RIGHT,TOP,BOTTOM
import tkinter as tk 
import speech_recognition as sr
import win32com.client as wincl

#Calling Class for chat prediction
ob = chatbot()

#main display chat window 
window = Tk()
window.title("Interview ChatBot")
window.geometry('550x450')

#top frame to display the chat history
frame1 = Frame(window, class_="TOP")
frame1.pack(expand=True, fill=BOTH)

#text area with scroll bar
textarea = Text(frame1, state=DISABLED)
vsb = Scrollbar(frame1, takefocus=
                0, command=textarea.yview)
vsb.pack(side=RIGHT, fill=Y)
textarea.pack(side=RIGHT, expand=YES, fill=BOTH)
textarea["yscrollcommand"]=vsb.set

#bottom frame to display current user question text box 
frame2 = Frame(window, class_="Chatbox_Entry")
frame2.pack(fill=X, anchor=N)

lbl = Label(frame2, text="User : ")
lbl.pack(side=LEFT)
 
button = tk.Button(window, text='Speak Now', width=25, command= lambda: listen()) 
button.place(relx=0.3, rely=0.9, anchor="c") 

button = tk.Button(window, text='Exit', width=25, command=window.destroy) 
button.place(relx=0.7, rely=0.9, anchor="c") 


def listen():
	# Record Audio
	r = sr.Recognizer()
	with sr.Microphone() as source:
		audio = r.listen(source)
	
	try:
		res = r.recognize_google(audio)
		ans = ob.test_output(res)
		pr="User : " + res + "\n" + "Bot : " + ans + "\n\n"
		textarea.config(state=NORMAL)
		textarea.insert(END,pr)
		textarea.config(state=DISABLED)
		
		speak=wincl.Dispatch("SAPI.SpVoice")
		speak.Speak(ans)
	except sr.UnknownValueError:
		print("Google Speech Recognition could not understand audio")
	except sr.RequestError as e:
		print("Could not request results from Google Speech Recognition service; {0}".format(e))
		
	

def bind_entry(self, event, handler):
    txt.bind(event, handler)

def clicked(event): 
    #to automate the scrollbar action downward according to the text
    relative_position_of_scrollbar = vsb.get()[1]
    res =txt.get() 
    #function call
    ans = ob.test_output(res)
    pr="User : " + res + "\n" + "Bot : " + ans + "\n\n"
    #the state of the textarea is normalto write the text to the top area in the interface
    textarea.config(state=NORMAL)
    textarea.insert(END,pr)
    #it is again disabled to avoid the user modifications in the history
    textarea.config(state=DISABLED)
    txt.delete(0,END)
    if relative_position_of_scrollbar == 1:
        textarea.yview_moveto(1)
    txt.focus()

# print("chk3")
txt = Entry(frame2,width=70)
txt.pack(side=LEFT,expand=YES, fill=BOTH)
txt.focus()
txt.bind("<Return>", clicked)

window.mainloop()