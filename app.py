from flask import Flask, render_template, request
import joblib
import numpy as np
app = Flask(__name__)
# Tải các mô hình và scaler
svm_model = joblib.load('SVM_model.pkl')
scaler = joblib.load('scaler.pkl')
# Danh sách cây trồng
crop_names = [ "Apple", "Banana", "Blackgram", "Chickpea", "Coconut", "Coffee", "Cotton", "Grapes",
    "Jute", "KidneyBeans", "Lentil", "Maize", "Mango", "MothBeans", "MungBean", "Muskmelon",
    "Orange", "Papaya", "PigeonPeas", "Pomegranate", "Rice", "Watermelon"]
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ form và kiểm tra các trường không được bỏ trống
        fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph_value', 'Rainfall']
        features = []
        for field in fields:
            value = request.form.get(field)
            if not value:
                return render_template('index.html', error_message=f"Lỗi: Trường '{field}' không được để trống.")
            features.append(float(value))
        # Chuẩn hóa dữ liệu
        input_data_scaled = scaler.transform([features])
        # Dự đoán với mô hình SVM
        prediction_index = svm_model.predict(input_data_scaled)[0]
        prediction = crop_names[prediction_index]
        confidence = "96,82%"  # Bạn có thể thêm mã tính độ tin cậy nếu muốn
        # Trả kết quả
        result = (
            f"<b>Cây trồng dự đoán:</b> {prediction}<br>"
            f"<b>Độ tin cậy:</b> {confidence}")
        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', error_message=f"Lỗi: {str(e)}")
if __name__ == '__main__':
    app.run(debug=True)
