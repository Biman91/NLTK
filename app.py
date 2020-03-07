from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


# load model
filename = 'classifier.pkl'
classifier = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = vectorizer.fit_transform(data).toarray()
		my_prediction = classifier.predict(vect)
	return render_template('output.html', prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)
