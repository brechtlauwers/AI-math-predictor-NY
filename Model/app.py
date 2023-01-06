# imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR

import pickle
import gradio as gr

pipeline = pickle.load(open("model.pkl", 'rb'))


def predict_score(grade, year, n_lvl1, n_lvl2, n_lvl3, n_lvl4, DBN):
    n_tested = n_lvl1 + n_lvl2 + n_lvl3 + n_lvl4

    pct_lvl1 = 0 if n_lvl1<=0 else (n_lvl1 / n_tested) * 100
    pct_lvl2 = 0 if n_lvl2<=0 else (n_lvl2 / n_tested) * 100
    pct_lvl3 = 0 if n_lvl3<=0 else (n_lvl3 / n_tested) * 100
    pct_lvl4 = 0 if n_lvl4<=0 else (n_lvl4 / n_tested) * 100

    district_n = DBN[:2]
    borough_n = DBN[2]
    encoding = {'M':0, 'X':1, 'K':2, 'Q':3, 'R':4}
    borough_n = encoding[borough_n]
    school_n = DBN[3:]

    X = {"Grade":grade, "Year":year, "Number Tested":n_tested, "Num Level 1":n_lvl1, "Pct Level 1":pct_lvl1,
         "Num Level 2":n_lvl2, "Pct Level 2":pct_lvl2, "Num Level 3":n_lvl3, "Pct Level 3":pct_lvl3,
         "Num Level 4":n_lvl4, "Pct Level 4":pct_lvl4, "District Number":district_n,
         "Borough Number":borough_n, "School Number":school_n}

    y = pipeline.predict(pd.DataFrame(X, index=[0]))
    return int(y[0])


#create input and output objects
grade = gr.Dropdown([3, 4, 5, 6, 7, 8], label="Grade")
year = gr.Dropdown([2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016], label="Year")
n_lvl1 = gr.Number(precision=0, label="Amount of students in Level 1")
n_lvl2 = gr.Number(precision=0, label="Amount of students in Level 2")
n_lvl3 = gr.Number(precision=0, label="Amount of students in Level 3")
n_lvl4 = gr.Number(precision=0, label="Amount of students in Level 4")
DBN = gr.Textbox(label="DBN (e.g. '01M188')")

#output object
output = gr.Textbox(label="Predicted Mean Scale Score")

gui = gr.Interface(fn=predict_score,
                   inputs=[grade, year, n_lvl1, n_lvl2, n_lvl3, n_lvl4, DBN],
                   outputs=[output])
gui.launch()
