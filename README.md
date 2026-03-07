# Phân loại Thể loại Âm nhạc 🎵🤖

Dự án này là một ứng dụng web giúp phân loại các tệp âm thanh thành 8 thể loại âm nhạc khác nhau, sử dụng Mạng nơ-ron tích chập 2 giai đoạn (2-Stage CNN) được xây dựng với TensorFlow/Keras và trích xuất lượng đặc trưng âm thanh thông qua thư viện `librosa`.

## Các chức năng chính

- **Dự đoán thể loại**: Tải lên một tệp âm thanh và nhận kết quả dự đoán thể loại cùng với độ tin cậy cho toàn bộ 8 danh mục.
- **Trực quan hóa**:
  - Tạo và hiển thị **Mel-Spectrogram** của âm thanh được tải lên.
  - Tạo và hiển thị biểu đồ **chuỗi đặc trưng âm thanh (Audio Features)**.
- **Các thể loại hỗ trợ**: Electronic (Điện tử), Experimental (Thể nghiệm), Folk (Dân gian), Hip-Hop, Instrumental (Không lời), International (Quốc tế), Pop, Rock.

## Cấu trúc dự án

- `app.py`: Mã nguồn ứng dụng máy chủ Flask chính. Xử lý âm thanh, dự đoán từ mô hình và xuất biểu đồ.
- `index.html`, `style.css`, `script.js`: Các tệp giao diện frontend.
- `requirements.txt`: Các thư viện Python cần thiết dùng để cấu hình tự động.
- `Mohinh/`: Thư mục lưu trữ mô hình Keras đã huấn luyện.
- `scaler_60_features_fixed.pkl`: Bộ Scikit-learn MinMaxScaler dùng để chuẩn hóa các đặc trưng âm thanh đã trích xuất.
- Các tệp `.ipynb`: Các Jupyter notebook được sử dụng để huấn luyện và đánh giá mô hình.

## Notebooks Huấn luyện
- Bạn có thể tham khảo chi tiết quy trình huấn luyện và xây dựng mô hình tại các liên kết dưới đây:
Google Colab (Huấn luyện mô hình thực tế): Tại đây
- Kaggle Notebook (Music Genre Classification - FMA Small): Tại đây
- Kaggle Notebook (Music Genre Classification - Spectrogram): Tại đây

## Hướng dẫn cài đặt

1. **Clone kho (Repository)** dự án xuống máy hoặc mở thư mục dự án lên.
2. **Tạo môi trường ảo** (khuyên dùng):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Trên Windows
   source .venv/bin/activate  # Trên macOS/Linux
   ```
3. **Cài đặt các thư viện phụ thuộc**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Lưu ý: Dự án này phụ thuộc lớn vào TensorFlow, Keras, Flask và Librosa)*

## Cấu hình

Trước khi chạy ứng dụng, hãy đảm bảo các đường dẫn tuyệt đối trong `app.py` cho mô hình và scaler trỏ chính xác đến các vị trí trên máy của bạn.

Mở file `app.py` và kiểm tra / cập nhật:
```python
MODEL_PATH = r'D:\Python_mohinh\Mohinh\KQ_Thu_Nghiem_8\MODEL_2STAGE_v10_Fusion_FMA_ONLY_Chunks3_FINAL.keras'
SCALER_PATH = r'D:\Python_mohinh\scaler_60_features_fixed.pkl'
```

## Chạy Ứng dụng

Khởi động máy chủ web Flask thông qua Terminal:
```bash
python app.py
```

Ứng dụng sẽ bắt đầu ở cổng `5050`. Bạn mở trình duyệt web lên và truy cập:
[http://localhost:5050](http://localhost:5050) hoặc [http://127.0.0.1:5050](http://127.0.0.1:5050)

## Web API (Các Endpoints)

- `GET /` : Trả về giao diện chính của Web.
- `POST /predict` : Nhận tệp âm thanh tải lên, trả về JSON với các dự đoán thể loại có được.
- `POST /spectrogram` : Nhận tệp âm thanh tải lên, trả về file ảnh `.png` của Mel-Spectrogram.
- `POST /audio_features` : Nhận tệp âm thanh tải lên, trả về file ảnh `.png` chụp lại chuỗi đặc trưng vừa được trích xuất.

## Công nghệ sử dụng

- **Backend**: Python, Flask
- **Học máy (Machine Learning)**: TensorFlow, Keras, Scikit-learn
- **Xử lý âm thanh**: Librosa
- **Xử lý mảng & Vẽ biểu đồ**: NumPy, Matplotlib
- **Frontend**: HTML5, CSS3, JavaScript
