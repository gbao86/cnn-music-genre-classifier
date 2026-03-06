// === (!!! MỚI !!!) Thêm fileListDisplay vào danh sách ===
const form = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const loader = document.getElementById('loader');
const loaderText = document.getElementById('loader-text');
const submitButton = document.getElementById('submit-button');
const resultsArea = document.getElementById('results-area');
const fileListDisplay = document.getElementById('file-list-display'); // <--- MỚI
const MAX_FILES = 12; // Giống backend

// === (!!! MỚI !!!) LẮNG NGHE SỰ KIỆN CHỌN FILE ===
fileInput.addEventListener('change', () => {
    const files = fileInput.files;
    
    // Xóa nội dung cũ
    fileListDisplay.innerHTML = ''; 

    if (files.length === 0) {
        fileListDisplay.style.display = 'none'; // Ẩn khung nếu không chọn file
        return;
    }

    // Hiển thị khung
    fileListDisplay.style.display = 'block';

    // Tạo tiêu đề
    const title = document.createElement('h4');
    title.innerText = `Đã chọn ${files.length} file:`;
    fileListDisplay.appendChild(title);

    // Tạo danh sách (ul)
    const list = document.createElement('ul');
    
    for (let i = 0; i < files.length; i++) {
        const item = document.createElement('li');
        item.innerText = files[i].name;
        list.appendChild(item);
        
        // Cảnh báo nếu chọn quá 12 file
        if (i >= (MAX_FILES - 1) && files.length > MAX_FILES) {
            const warning = document.createElement('p');
            warning.className = 'file-limit-warning';
            warning.innerText = `... và ${files.length - MAX_FILES} file khác. (Chỉ ${MAX_FILES} file đầu tiên sẽ được xử lý!)`;
            list.appendChild(warning);
            break; // Ngừng lặp
        }
    }
    
    fileListDisplay.appendChild(list);
});


// === LOGIC SUBMIT FORM (Giữ nguyên) ===
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const files = fileInput.files;
    if (files.length === 0) return alert("Vui lòng chọn ít nhất 1 file MP3!");
    if (files.length > MAX_FILES) {
        alert(`Bạn đã chọn ${files.length} file. Chỉ ${MAX_FILES} file đầu tiên sẽ được xử lý.`);
    }

    // --- 1. Reset UI ---
    submitButton.disabled = true;
    submitButton.innerHTML = 'Đang xử lý...';
    loaderText.innerText = `Đang xử lý ${Math.min(files.length, MAX_FILES)} file...`;
    loader.style.display = 'block';
    resultsArea.innerHTML = ''; // Xóa kết quả cũ

    // --- 2. Tạo FormData (Chỉ lấy tối đa MAX_FILES) ---
    const formData = new FormData();
    for (let i = 0; i < Math.min(files.length, MAX_FILES); i++) {
        formData.append('file', files[i]);
    }

    try {
        // --- 3. GỌI /predict ---
        const predictRes = await fetch('/predict', { method: 'POST', body: formData });
        
        const resultsList = await predictRes.json();
        
        if (!predictRes.ok) {
            throw new Error(resultsList.error || 'Lỗi không xác định từ server');
        }

        // --- 4. Lặp qua danh sách kết quả và tạo HTML ---
        resultsList.forEach((result, index) => {
            const card = createResultCard(result, index);
            resultsArea.appendChild(card);
            createChart(result.all_scores, `chart-${index}`);
        });

    } catch (err) {
        alert("Lỗi nghiêm trọng: " + err.message);
        console.error(err);
    } finally {
        // --- 5. Hoàn tất ---
        submitButton.disabled = false;
        submitButton.innerHTML = 'Phân loại';
        loader.style.display = 'none';
    }
});

/**
 * Tạo một thẻ (card) HTML cho một kết quả dự đoán
 * (Hàm này giữ nguyên)
 */
function createResultCard(result, index) {
    const cardDiv = document.createElement('div');
    cardDiv.className = 'result-card glass';
    
    cardDiv.innerHTML = `
        <h3>${result.file_name}</h3>
        <div class="result-main-info">
            <h2>${result.genre_du_doan}</h2>
            <p>Độ tin cậy: ${result.do_tin_cay}</p>
        </div>
        <div class="chart-container">
            <canvas id="chart-${index}"></canvas>
        </div>
        <div class="image-toggle-buttons">
            <button class="toggle-btn" data-type="spectrogram" data-index="${index}">Xem Quang phổ</button>
            <button class="toggle-btn" data-type="features" data-index="${index}">Xem 57 Đặc trưng</button>
        </div>
        <div id="image-container-${index}" class="image-container">
            </div>
    `;
    return cardDiv;
}

/**
 * Vẽ biểu đồ Chart.js
 * (Hàm này giữ nguyên)
 */
function createChart(scores, canvasId) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const labels = Object.keys(scores);
    const values = Object.values(scores).map(v => (v * 100).toFixed(2));

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Độ tin cậy (%)',
                data: values,
                backgroundColor: 'rgba(0, 219, 222, 0.6)',
                borderColor: '#00dbde',
                borderWidth: 2,
                borderRadius: 6,
                maxBarThickness: 40
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: ctx => `${ctx.raw}%` } }
            },
            scales: {
                y: {
                    beginAtZero: true, max: 100,
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#fff' }
                },
                x: { grid: { display: false }, ticks: { color: '#fff' } }
            }
        }
    });
}

/**
 * Xử lý các nút nhấn "Xem Quang phổ" và "Xem Đặc trưng"
 * (Hàm này giữ nguyên)
 */
resultsArea.addEventListener('click', async (e) => {
    if (!e.target.classList.contains('toggle-btn')) return;

    const button = e.target;
    const dataType = button.dataset.type;
    const index = button.dataset.index;

    // Lấy đúng file (chỉ lấy trong số 12 file đầu)
    const file = fileInput.files[index];
    if (!file) return;

    const imageContainer = document.getElementById(`image-container-${index}`);
    const allButtons = button.parentElement.querySelectorAll('.toggle-btn');
    
    // UI: Hiển thị "Đang tải..."
    imageContainer.style.display = 'block';
    imageContainer.innerHTML = '<div class="spinner" style="margin: 10px auto;"></div>';
    allButtons.forEach(btn => btn.disabled = true);
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const endpoint = (dataType === 'spectrogram') ? '/spectrogram' : '/audio_features';
        
        const res = await fetch(endpoint, { method: 'POST', body: formData });
        if (!res.ok) throw new Error('Server không thể tạo ảnh.');

        const blob = await res.blob();
        const imgUrl = URL.createObjectURL(blob);
        
        imageContainer.innerHTML = `<img src="${imgUrl}" alt="${dataType} image">`;
        
        allButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');

    } catch (err) {
        imageContainer.innerHTML = `<p style="color: red;">Lỗi: ${err.message}</p>`;
    } finally {
        allButtons.forEach(btn => btn.disabled = false);
    }
});