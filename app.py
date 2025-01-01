from flask import Flask, render_template, request
import joblib
import pandas as pd
app = Flask(__name__)
# Tải mô hình và các công cụ xử lý
try:
    svm_model = joblib.load('SVM_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError as e:
    print(f"Lỗi: Không tìm thấy file mô hình hoặc scaler. {str(e)}")
    exit()
# Độ chính xác mô hình (thêm thủ công hoặc lấy từ huấn luyện)
MODEL_ACCURACY = 96,82  # Độ chính xác giả định
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None; error_message = None
    if request.method == "POST":
        try:
            # Lấy dữ liệu từ form
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])
            # Đóng gói dữ liệu vào DataFrame
            input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
            # Chuẩn hóa dữ liệu
            input_data_scaled = scaler.transform(input_data)
            # Dự đoán cây trồng
            svm_prediction_index = svm_model.predict(input_data_scaled)[0]
            prediction = label_encoder.inverse_transform([svm_prediction_index])[0]
        except ValueError:
            error_message = "Vui lòng nhập đầy đủ và đúng định dạng các thông số!"
        except Exception as e:
            error_message = f"Lỗi không xác định: {str(e)}"
    return render_template("index.html", prediction=prediction, error_message=error_message, accuracy=MODEL_ACCURACY)
if __name__ == "__main__":
    app.run(debug=True)
