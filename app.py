from flask import Flask, render_template
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

# Hàm để phân cụm dữ liệu
def cluster_weather_data():
    # Tải dữ liệu
    data_weather = pd.read_csv('data_weather.csv')
    
    # Chuyển đổi cột 'date' thành định dạng datetime
    if 'datetime' in data_weather.columns:
        data_weather['datetime'] = pd.to_datetime(data_weather['datetime'])
    
    # Chọn các cột để phân cụm
    # X = data_weather[['tempmax', 'tempmin', 'humidity']]
    X = data_weather[['temp']]
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Tạo mô hình K-means
    kmeans = KMeans(n_clusters=3, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Thêm kết quả vào DataFrame
    data_weather['Cluster'] = clusters
    return data_weather

# Hàm để vẽ biểu đồ
def plot_weather_data(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['tempmax'], data['tempmin'], c=data['Cluster'], cmap='viridis', marker='o')
    plt.title('Kết quả phân cụm nhiệt độ dựa trên mô hình K-means Clustering')
    plt.colorbar(label='3 cụm đã được phân cụm')
    
    plt.savefig('static/weather_plot.png')  # Lưu biểu đồ vào thư mục static
    plt.close()  # Đóng biểu đồ để không hiển thị trên máy chủ

# Hàm để dự đoán nhiệt độ cho 7 ngày tiếp theo
def predict_weather(data, start_date='2024-11-01'):
    features = ['tempmax', 'tempmin', 'humidity']
    target_tempmax = 'tempmax'
    target_tempmin = 'tempmin'

    # Kiểm tra xem các cột cần thiết có tồn tại hay không
    if not all(col in data.columns for col in features + [target_tempmax, target_tempmin]):
        print("Dữ liệu không đầy đủ để dự đoán.")
        return []

    # Tạo mô hình hồi quy rừng ngẫu nhiên cho tempmax và tempmin
    model_tempmax = RandomForestRegressor(n_estimators=100, random_state=42,oob_score=True)
    model_tempmin = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)

    # Chia dữ liệu thành tập huấn luyện
    X = data[features]
    y_tempmax = data[target_tempmax]
    y_tempmin = data[target_tempmin]

    # Huấn luyện mô hình
    model_tempmax.fit(X, y_tempmax)
    model_tempmin.fit(X, y_tempmin)

    # Dự đoán cho 7 ngày tiếp theo
    predictions = []

    # Kiểm tra kiểu dữ liệu cột datetime
    if data['datetime'].dtype == 'float64':
        data['datetime'] = pd.to_datetime(data['datetime'], unit='s', origin='unix')
    elif data['datetime'].dtype != 'datetime64[ns]':
        print("Cột datetime không có kiểu dữ liệu hợp lệ.")
        return []

    last_row = data.iloc[-1]  # Lấy dòng cuối cùng của dữ liệu

    # Thay đổi ngày bắt đầu
    start_date = pd.to_datetime(start_date)
    
    for i in range(1, 8):  # Dự đoán cho 7 ngày tiếp theo
        next_date = start_date + pd.Timedelta(days=i)

        # Tạo dữ liệu mới để dự đoán với một chút nhiễu
        new_data = pd.DataFrame({
            'tempmax': [last_row['tempmax'] + np.random.uniform(-2, 2)],
            'tempmin': [last_row['tempmin'] + np.random.uniform(-2, 2)],
            'humidity': [last_row['humidity'] + np.random.uniform(-10, 10)]
        })

        # Dự đoán nhiệt độ tối đa và tối thiểu
        predicted_tempmax = model_tempmax.predict(new_data)[0]
        predicted_tempmin = model_tempmin.predict(new_data)[0]

        # Tính nhiệt độ trung bình
        predicted_temp = (predicted_tempmax + predicted_tempmin) / 2

        predictions.append((next_date.strftime('%Y-%m-%d'), predicted_tempmax, predicted_tempmin, predicted_temp))

        # In ra kết quả dự đoán chi tiết
        print(f"Dự đoán cho ngày {next_date.strftime('%Y-%m-%d')}: "
              f"Nhiệt độ tối đa = {predicted_tempmax:.2f}, "
              f"Nhiệt độ tối thiểu = {predicted_tempmin:.2f}, "
              f"Nhiệt độ trung bình = {predicted_temp:.2f}")

    print("Kết quả dự đoán:", predictions)
    oob_score_tempmax=model_tempmax.oob_score_
    oob_score_tempmin=model_tempmin.oob_score_
    print(oob_score_tempmax,oob_score_tempmin )
    return predictions

@app.route('/')
def index():
    # Phân cụm dữ liệu
    data_weather = cluster_weather_data()
    
    # Vẽ biểu đồ
    plot_weather_data(data_weather)
    
    # Dự đoán thời tiết
    predictions = predict_weather(data_weather)
    
    # Trả về template với dữ liệu dự đoán
    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
