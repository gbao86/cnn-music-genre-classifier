import os
import io
import pickle
import traceback
from flask import send_file
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory
import librosa
import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg') # Backend không GUI để chạy trên server
import matplotlib.pyplot as plt
import gc
from sklearn.preprocessing import MinMaxScaler 

warnings.filterwarnings('ignore')

# === CẤU HÌNH ===
print(">>> Cấu hình TensorFlow...")
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e: print(f"Lỗi GPU: {e}")

# === LOAD MODEL & SCALER ===
print(">>> Đang tải Resources...")
# !!! ĐƯỜNG DẪN TUYỆT ĐỐI !!!
MODEL_PATH = r'D:\Python_mohinh\Mohinh\KQ_Thu_Nghiem_8\MODEL_2STAGE_v10_Fusion_FMA_ONLY_Chunks3_FINAL.keras'
SCALER_PATH = r'D:\Python_mohinh\scaler_60_features_fixed.pkl'

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(">>> Load Model OK!")
except Exception as e: print(f"!!! LỖI MODEL: {e}")

scaler = None
try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f">>> Load Scaler OK! (Mean shape: {scaler.mean_.shape})")
except Exception as e:
    print(f"!!! LỖI SCALER: {e}")
    scaler = None

# CONSTANTS
NUM_AUDIO_FEATURES = 60
class_names = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']
IMG_HEIGHT = 128; IMG_WIDTH = 431 
DURATION = 10; HOP_LENGTH = 512; N_MFCC = 20
SAMPLES_NEEDED = 220500
ADJUSTMENT_WEIGHTS = np.array([1.0] * 8)

# Warm-up
try:
    model.predict([np.zeros((1, 128, 431, 3)), np.zeros((1, 60))])
    print(">>> Warm-up OK!")
except: pass

# ================== CÁC HÀM XỬ LÝ ==================
def read_audio(path):
    try:
        y, sr = librosa.load(path, mono=True, sr=22050, duration=DURATION)
        if len(y) < SAMPLES_NEEDED: y = np.pad(y, (0, SAMPLES_NEEDED - len(y)))
        else: y = y[:SAMPLES_NEEDED]
        return y, 22050
    except Exception as e:
        print(f"Lỗi đọc audio: {e}")
        return None, None

def create_spectrogram_array(y, sr):
    try:
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=IMG_HEIGHT, hop_length=HOP_LENGTH, fmax=8000)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        if spec_db.shape[1] < IMG_WIDTH: spec_db = np.pad(spec_db, ((0,0), (0, IMG_WIDTH-spec_db.shape[1])))
        else: spec_db = spec_db[:, :IMG_WIDTH]
        return np.stack((spec_db,)*3, axis=-1).astype("float32")
    except: return None

def extract_vector_60(y, sr):
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)
        roll = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
        cont = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LENGTH)
        flat = librosa.feature.spectral_flatness(y=y, hop_length=HOP_LENGTH)
        cens = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=HOP_LENGTH)
        poly = librosa.feature.poly_features(y=y, sr=sr, order=2, hop_length=HOP_LENGTH)
        
        feats = np.vstack([mfcc, chroma, cent, bw, roll, zcr, rms, cont, flat, cens, poly])
        vec_raw = np.mean(feats, axis=1)
        
        if scaler:
            return scaler.transform(vec_raw.reshape(1, -1)).flatten().astype("float32")
        return vec_raw.astype("float32")
    except: return None

# Hàm vẽ Sequence (cho route /audio_features)
def extract_sequence_for_plot(y, sr):
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
        # ... (Lấy đại diện vài feature để vẽ cho nhẹ)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
        cont = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LENGTH)
        
        feats = np.vstack([mfcc, chroma, cent, cont]) # Vẽ ít thôi cho đẹp
        
        # Transpose để trục hoành là thời gian
        return feats.T 
    except: return None

# ================== FLASK APP ==================
app = Flask(__name__)

@app.route("/")
def index(): return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename): return send_from_directory('.', filename)

# API: DỰ ĐOÁN
@app.route("/predict", methods=['POST'])
def predict():
    files = request.files.getlist('file')
    if not files: return jsonify({"error": "No file"}), 400
    
    imgs, vecs, names, paths = [], [], [], []
    
    try:
        for i, f in enumerate(files):
            path = f"temp_{i}.mp3"
            f.save(path)
            paths.append(path)
            
            y, sr = read_audio(path)
            if y is None: continue
            
            img = create_spectrogram_array(y, sr)
            vec = extract_vector_60(y, sr)
            
            if img is not None and vec is not None:
                imgs.append(img)
                vecs.append(vec)
                names.append(f.filename)
        
        if not imgs: return jsonify({"error": "Failed all files"}), 500
        
        preds = model.predict([np.array(imgs), np.array(vecs)])
        
        results = []
        for i, prob in enumerate(preds):
            prob = prob * ADJUSTMENT_WEIGHTS
            prob = prob / np.sum(prob)
            idx = np.argmax(prob)
            results.append({
                "file_name": names[i],
                "genre_du_doan": class_names[idx],
                "do_tin_cay": f"{np.max(prob)*100:.2f}%",
                "all_scores": {k: float(v) for k,v in zip(class_names, prob)}
            })
        return jsonify(results)
        
    except Exception as e: 
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        for p in paths: 
            if os.path.exists(p): os.remove(p)
        gc.collect()

# API: ẢNH SPECTROGRAM (Bị thiếu ở code cũ -> Đã thêm lại)
@app.route("/spectrogram", methods=['POST'])
def spectrogram():
    f = request.files.getlist('file')[0]
    path = "temp_spec.mp3"
    try:
        f.save(path)
        y, sr = read_audio(path)
        
        # Vẽ ảnh
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        
        fig = plt.figure(figsize=(10, 3))
        plt.imshow(spec_db, aspect='auto', origin='lower', cmap='magma')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        return send_file(buf, mimetype='image/png')
    except Exception as e: return jsonify({"error": str(e)}), 500
    finally: 
        if os.path.exists(path): os.remove(path)

# API: ẢNH ĐẶC TRƯNG
@app.route("/audio_features", methods=['POST'])
def features_plot():
    f = request.files.getlist('file')[0]
    path = "temp_feat.mp3"
    try:
        f.save(path)
        y, sr = read_audio(path)
        feats = extract_sequence_for_plot(y, sr)
        
        # Chuẩn hóa
        scaler_plot = MinMaxScaler()
        feats_norm = scaler_plot.fit_transform(feats).T
        
        fig = plt.figure(figsize=(10, 3))
        plt.imshow(feats_norm, aspect='auto', origin='lower', cmap='viridis')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        return send_file(buf, mimetype='image/png')
    except Exception as e: return jsonify({"error": str(e)}), 500
    finally: 
        if os.path.exists(path): os.remove(path)

if __name__ == "__main__":
    print(">>> Đang khởi động Server trên cổng 5050...")
    app.run(debug=True, use_reloader=False, port=5050, host='0.0.0.0')