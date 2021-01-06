from tkinter import *
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import pickle
from sklearn.model_selection import train_test_split
import xgboost
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

with open('Saved/countVectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

model = xgboost.XGBClassifier()

model.load_model("Saved/xgboost_bestModel.json")

def detect_button(TEXT, model, vectorizer, lemmatizer, pred_button):
    txt = TEXT.get("1.0",'end-1c')
    print(txt)
    t = re.sub(r'\W', ' ', str(txt))
    t = re.sub(r'\s+[a-zA-Z]\s+', ' ', t)
    t = re.sub(r'\^[a-zA-Z]\s+', ' ', t)
    t = re.sub(r'\s+', ' ', t, flags=re.I)
    t = re.sub(r'^b\s+', '', t)
    t = t.lower()
    t = t.split()
    t = [lemmatizer.lemmatize(word) for word in t]
    t = ' '.join(t)

    x_test = vectorizer.transform([t]).toarray()
    print(x_test.shape)
    [pre] = model.predict(x_test)
    if pre==1:
        pred_button.config(bg='red', text='TOXIC')
    if pre==0:
        pred_button.config(bg='green', text='SAFE')
    TEXT.clipboard_clear()

    return

window = Tk()

window.title("Toxic Commend Detector")
window.geometry("490x400")

text = Text(window,height=15,width=60)
text.place(x=2,y=2)

pred_button = Button(window, text='', bg='white', width=10, height=5)
pred_button.place(x=400,y=300)

run_button = Button(window, text='Detect', command=lambda :detect_button(text, model, vectorizer,lemmatizer,pred_button), width=10, height=5)
run_button.place(x=10,y=300)





window.mainloop()




