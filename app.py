from flask import Flask
import pickle
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

model=pickle.load('model.pkl','rb')
cv=CountVectorizer()

def remove_pun(msg):
    nonPunc = [char for char in msg if char not in string.punctuation]
    nonPunc = "".join(nonPunc)
    return nonPunc

app = Flask(__name__)

@app.route('/predict/<msg>',methods=['GET','POST'])
def predict(msg):
    df = pd.read_csv("SMSSpamCollection", sep='\t', names=['class', 'message'])
    df['message'] = df['message'].apply(remove_pun)
    xSet = df['message'].values
    ySet = df['class'].values

    x_train, x_test, y_train, y_test = train_test_split(xSet, ySet, test_size=0.25, random_state=0)
    v = cv.fit_transform(x_train)
    v=cv.transform([msg])
    predict=model.predict(v)
    if predict[0]==1:
        return "True"
    else:
        return "False"




if __name__ == "__main__":
    app.run(debug=True)
