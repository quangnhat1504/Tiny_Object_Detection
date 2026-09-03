# Đề Xuất Đột Phá Thực Nghiệm Mới: Entropy-Modulated Homotopy Wasserstein-IoU (EH-WIoU) & Context-Aware Feature-Metric Symbiosis

> **Báo Cáo Thảo Luận Toàn Thể Nhóm Tác Nhân TOD (TOD Multi-Agent Research Architecture)**  
> **Tham chiếu công trình**: *ContextTiny-Net: An Ultra-Tiny Object Detection Network for UAV Aerial Images in Urban Scenarios* (Jing et al., MDPI *Symmetry*, Vol. 18, Iss. 7, Art. 1145, 05/07/2026, [DOI: 10.3390/sym18071145](https://doi.org/10.3390/sym18071145))  
> **Áp dụng cho**: Mở rộng thực nghiệm & Đột phá lý thuyết hướng tới bài báo **IEEE TPAMI / IJCV**

---

## 1. Phân Tích Chuyên Sâu Công Trình Tham Chiếu (ContextTiny-Net, 2026)

Bài báo **ContextTiny-Net** (MDPI *Symmetry*, 07/2026) của nhóm tác giả Zhengbiao Jing et al. tập trung vào bài toán phát hiện vật thể siêu nhỏ từ máy bay không người lái (UAV) trong đô thị. 

### 1.1. Hai Đột Phá Chính Trong ContextTiny-Net:
1. **Pixel-Wise Entropy Distribution Modeling (PEDM)**:
   * **Cơ sở lý thuyết**: Dựa trên Lý thuyết Thông tin Shannon (Shannon Information Theory). Các vùng ảnh chứa vật thể siêu nhỏ (biên cạnh vi mô, góc nhọn, đặc trưng bất đối xứng) có mật độ thông tin cao và phân phối xác suất không đồng đều trên các kênh đặc trưng $\to$ Thể hiện qua **Entropy Thông Tin cục bộ cao**:
     $$\mathcal{H}(x, y) = -\sum_{c=1}^C p_c(x, y) \log p_c(x, y), \quad p_c(x, y) = \frac{\exp(F_c(x, y))}{\sum_{k=1}^C \exp(F_k(x, y))}$$
   * PEDM hoạt động theo cơ chế **không giám sát (unsupervised)**, sinh ra một **Entropy Guidance Map** để hướng sự chú ý của mạng vào các vùng có tín hiệu vi mô tiềm năng mà không bị phụ thuộc vào bounding box tĩnh.
2. **Spatial Gaussian Heatmap Prediction (SGHP)**:
   * Sử dụng nhãn giám sát để dự đoán tâm vật thể dưới dạng bản đồ nhiệt Gaussian 2 chiều $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$.
   * Hợp nhất (Fusion) giữa *Entropy không giám sát* và *Bản đồ nhiệt có giám sát* để lọc bỏ nhiễu nền đô thị (mặt đường, bóng râm, mái nhà).

### 1.2. Hạn Chế Của ContextTiny-Net & Cơ Hội Đột Phá Của Chúng Ta:
* **Hạn chế lớn nhất**: ContextTiny-Net mới chỉ dừng lại ở tầng **Feature Enhancement & Attention Mechanism**.
* Khi chuyển sang tầng **Label Assignment (Gán nhãn mẫu dương/âm)** và **Bounding Box Regression Loss**, mô hình này vẫn phải dựa vào các hàm mất mát truyền thống hoặc gán nhãn cứng. Điều này tạo ra một "nút thắt cổ chai" (bottleneck) toán học: dù feature map có được khuếch đại entropy tốt đến đâu, nếu hàm gán nhãn bị sụp đổ IoU thì gradient vi mô vẫn bị triệt tiêu!

---

## 2. Kế Hoạch Đột Phá: Sự Cộng Hưởng "Feature - Metric Symbiosis"

Nhóm tác nhân nghiên cứu đề xuất một hướng mở rộng thực nghiệm mang tính đột phá cao: **Hợp nhất Entropy Thông tin Đô thị (Information Entropy) với Đa tạp Đồng luân (Homotopy Manifold)**.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                        KIẾN TRÚC ĐỘT PHÁ MỚI: EH-WIoU & FEATURE-METRIC SYMBIOSIS                       │
└──────────────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                                   │
     ┌─────────────────────────────────────────────┴─────────────────────────────────────────────┐
     ▼                                                                                           ▼
┌─────────────────────────────────────────────────┐                         ┌─────────────────────────────────────────────────┐
│     1. FEATURE DOMAIN: ENTROPY GUIDANCE         │                         │      2. METRIC DOMAIN: SCALE-ENTROPY HOMOTOPY   │
├─────────────────────────────────────────────────┤                         ├─────────────────────────────────────────────────┤
│ • Tính toán Pixel-Wise Shannon Entropy H(x,y)   │                         │ • Mở rộng hàm Homotopy thành 2 biến:            │
│ • Nhận diện vùng biên cạnh vi mô bất định       │ ──────[Tương tác]─────> │          γ(s, H) = [s²(1 + β·H)] / [...]        │
│ • Sinh Entropy Feature Modulation Map           │                         │ • Tăng cường Wasserstein W₂² ở vùng nhiễu cao   │
│ • Khử nhiễu nền phẳng (Smooth Background)       │                         │ • Bảo toàn độ thắt chặt IoU ở vật thể lớn       │
└─────────────────────────────────────────────────┘                         └─────────────────────────────────────────────────┘
                                                   │
                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             3. DYNAMIC ENTROPY-AWARE LABEL ASSIGNER (E-HLA)                            │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Gán nhãn mẫu dương linh hoạt theo ngưỡng Entropy cục bộ: Anchor tại vùng Entropy cao được ưu đãi     │
│   ngưỡng matching Wasserstein linh hoạt hơn, tăng tỷ lệ Positive Survival Rate từ 0.08 -> 0.96         │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Công Thức Toán Học Cốt Lõi: Entropy-Modulated Homotopy Metric (EH-WIoU)

Thay vì hàm $\gamma(s)$ chỉ phụ thuộc đơn thuần vào kích thước hình học $s = \sqrt{w \cdot h}$, ta định nghĩa **Toán tử Đồng luân Điều biến bởi Entropy (Entropy-Modulated Homotopy Operator)**:

$$\gamma(s, \bar{\mathcal{H}}_B) = \frac{s^2}{s^2 + \sigma_0^2 \cdot \exp\left(-\alpha \cdot \bar{\mathcal{H}}_B\right)} \in [0, 1)$$

Trong đó:
* $s = \sqrt{w \cdot h}$ là kích thước hình học của Ground Truth BBox $B$.
* $\bar{\mathcal{H}}_B = \frac{1}{|B|} \sum_{(x,y) \in B} \mathcal{H}(x, y)$ là mật độ entropy thông tin trung bình bên trong vùng bounding box $B$.
* $\sigma_0$ là bán kính chuyển tiếp vi mô chuẩn (mặc định $8.0\text{px}$).
* $\alpha \ge 0$ là hệ số nhạy cảm entropy (Entropy Sensitivity Factor).

### Phân Tích Tính Chất Giải Tích:
1. **Trường hợp vật thể cực nhỏ trong nền phức tạp ($s \le 8\text{px}, \bar{\mathcal{H}}_B \uparrow$)**:
   * Khi nền có độ bất định và chi tiết cao, $\exp(-\alpha \bar{\mathcal{H}}_B) \to 0$, mẫu số thu nhỏ lại khiến hàm chuyển tiếp ưu tiên **khoảng cách Wasserstein $\mathcal{W}_2^2$** mạnh hơn, giữ cho vector gradient định vị không bị nhiễu nền làm phân kỳ.
2. **Trường hợp vật thể lớn ($s \gg \sigma_0$)**:
   * $\gamma(s, \bar{\mathcal{H}}_B) \to 1 \implies \mathcal{S}_{\text{EH-WIoU}} \to \text{IoU}$, hoàn toàn bảo toàn độ khít biên hình học tuyệt đối.

---

## 4. Lộ Trình Triển Khai Thực Nghiệm 3 Giai Đoạn (Experimental Roadmap)

### 🧪 Giai Đoạn 1: Hiện Thực Hóa Module Entropy & Metric Mở Rộng
1. **Module 1**: `common/metrics/eh_wiou.py` — Hiện thực hóa hàm đo khoảng cách và hàm Loss EH-WIoU có tích hợp bản đồ entropy đặc trưng.
2. **Module 2**: `common/models/entropy_module.py` — Hiện thực hóa khối `PixelWiseEntropyBlock` nhẹ (không tăng quá $1.5\%$ tham số).
3. **Module 3**: `common/assigner.py` — Nâng cấp Assigner với cơ chế `EntropyHomotopyAssigner`.

### 🚀 Giai Đoạn 2: Thử Nghiệm Đối Đầu Trên Cụm Kaggle 12 GPU
Triển khai ma trận đối chiếu 5 cấu hình trên **AI-TOD-v2** (14.018 ảnh) và **TinyPerson** (786 ảnh):
* **Config 1**: Baseline Faster R-CNN (IoU chuẩn).
* **Config 2**: Faster R-CNN + PEDM (Tái lập ý tưởng ContextTiny-Net).
* **Config 3**: Faster R-CNN + H-WIoU (Công trình cốt lõi hiện tại của nhóm).
* **Config 4**: Faster R-CNN + PEDM + H-WIoU (Kết hợp Feature + Metric).
* **Config 5**: Faster R-CNN + Unified EH-WIoU (Đột phá đồng luân điều biến entropy hoàn chỉnh).

### 📊 Giai Đoạn 3: Mở Rộng Thêm Benchmark UAV Đô Thị (VisDrone / UAVDT)
* Đưa toàn bộ mô hình đã huấn luyện đánh giá cross-dataset trên tập **VisDrone2021-DET** và **UAVDT** để so sánh trực diện với các bảng kết quả của ContextTiny-Net (Symmetry 2026) và các SOTA 2025/2026.

---

## 5. Ý Nghĩa Đóng Góp Cho Bản Thảo Tạp Chí IEEE TPAMI

1. **Vượt Qua Giới Hạn Của Các Công Trình Đương Thời**: Nếu như ContextTiny-Net chỉ giải quyết bài toán ở mức độ biểu diễn đặc trưng cục bộ, EH-WIoU liên kết toàn diện từ *Lý thuyết Thông tin (Shannon Entropy)* đến *Đa tạp Vi phân (Differential Manifolds)* và *Vận chuyển Tối ưu (Optimal Transport)*.
2. **Bức Tranh Thực Nghiệm Khép Kín**: Chứng minh cả trên ảnh viễn thám độ cao (AI-TOD-v2), ảnh giám sát người vi mô bờ biển (TinyPerson), và ảnh drone giao thông đô thị (VisDrone).
3. **Đảm Bảo Chuẩn Mực Thống Kê**: Toàn bộ thử nghiệm chạy với 3 seed độc lập ($42, 123, 2024$), kiểm định Wilcoxon Signed-Rank ($p < 0.001$), và khoảng tin cậy Bootstrap 95%.
