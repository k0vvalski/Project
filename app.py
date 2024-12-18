from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        date = float(request.form['date'])
        day = int(request.form['day']) 
        period = float(request.form['period'])
        nswprice = float(request.form['nswprice'])
        nswdemand = float(request.form['nswdemand'])
        vicprice = float(request.form['vicprice'])
        vicdemand = float(request.form['vicdemand'])
        transfer = float(request.form['transfer'])

        input_data = np.array([[date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer]])

        if hasattr(model, 'predict'):
            prediction = model.predict(input_data)
            classs = 'UP' if prediction[0] == 1 else 'DOWN'
            return render_template('result.html', prediction=classs)
        else:
            raise Exception("Модель не поддерживает метод 'predict'.")
    except Exception as e:
        return render_template('index.html', prediction=f'Ошибка: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)