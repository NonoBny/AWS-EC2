import pickle
import pandas as pd
from flask import Flask, request

app = Flask(__name__)

with open('churn_model.pkl', 'rb') as f:
    clf = pickle.load(f)

df = pd.read_csv('cellular_churn_greece.csv', index_col=0)
X = df.drop('churned', axis=1)


@app.route('/predict_churn', methods=['GET'])
def predict():
    inputs = request.args.to_dict()
    data = pd.DataFrame(inputs, index=[0])
    inputs = data.to_numpy().astype(float)
    prediction = clf.predict(inputs)

    return str(prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
