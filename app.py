import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import RobustScaler
import pickle

app = Flask(__name__)
model = pickle.load(open('model2.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    rob_scaler = RobustScaler()
    final_features = rob_scaler.fit_transform(final_features)
    print(final_features)
    prediction = model.predict(final_features)
    print(prediction)
    output = round(prediction[0], 2)
    print(output)

    return render_template('index.html', prediction_text='Transaction frauduleuse {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)