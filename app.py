from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Загрузка модели
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получение данных из формы
        date = float(request.form['date'])
        day = int(request.form['day'])  # Изменено на int
        period = float(request.form['period'])
        nswprice = float(request.form['nswprice'])
        nswdemand = float(request.form['nswdemand'])
        vicprice = float(request.form['vicprice'])
        vicdemand = float(request.form['vicdemand'])
        transfer = float(request.form['transfer'])

        # Подготовка входных данных
        input_data = np.array([[date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer]])

        # Предсказание
        if hasattr(model, 'predict'):
            prediction = model.predict(input_data)
            classs = 'UP' if prediction[0] == 1 else 'DOWN'  # Предполагаем, что 1 - это 'UP', 0 - это 'DOWN'
            return render_template('result.html', prediction=classs)
        else:
            raise Exception("Модель не поддерживает метод 'predict'.")
    except Exception as e:
        return render_template('index.html', prediction=f'Ошибка: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)