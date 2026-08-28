# Lộ Trình Mở Rộng Nghiên Cứu: H-WIoU v2 & Các Hướng Đột Phá Tiếp Nối

Tài liệu này ghi nhận chi tiết cơ sở toán học, thiết kế kiến trúc và kết quả kiểm thử đơn vị của **3 Hướng Đột Phá Mở Rộng** cho bài toán Phát hiện Vật thể Siêu nhỏ (Tiny Object Detection - TOD).

---

## 1. Tổng Quan 3 Hướng Mở Rộng (H-WIoU v2 Extensions)

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │             KIẾN TRÚC H-WIOU v2 EXTENSIONS             │
                                  └──────────────────────────┬─────────────────────────────┘
                                                             │
            ┌────────────────────────────────────────────────┼────────────────────────────────────────┐
            ▼                                                ▼                                        ▼
┌──────────────────────────────────────┐ ┌──────────────────────────────────────┐ ┌──────────────────────────────────────┐
│ 1. Dynamic Uncertainty (DU-HWIoU)    │ │ 2. Spectral Wavelet (SW-HWIoU)       │ │ 3. Oriented Homotopy (O-HWIoU)       │
│ - Tự động học σ_0(z) theo từng vùng  │ │ - Khai thác năng lượng tần số cao    │ │ - Ánh xạ BBox xoay sang Gaussian 2D  │
│ - Điều biến độ cong đa tạp thích nghi│ │ - Bổ chính gradient vi mô từ DWT     │ │ - Phù hợp tập viễn thám AI-TOD-R     │
└──────────────────────────────────────┘ └──────────────────────────────────────┘ └──────────────────────────────────────┘
```

---

## 2. Chi Tiết Lý Thuyết & Triển Khai Toán Học

### Hướng 1: Dynamic Uncertainty-Aware Homotopy (DU-HWIoU)
* **File nguồn**: [common/metrics/dynamic_uncertainty_h_wiou.py](file:///c:/Users/ADMIN/_Project/tiny-object-detection/common/metrics/dynamic_uncertainty_h_wiou.py)
* **Cơ sở toán học**:
  Thay vì cố định tham số ngưỡng $\sigma_0$ trên toàn bộ tập dữ liệu, mô-đun `UncertaintyHomotopyPredictor` sử dụng mạng 2 tầng MLP nhẹ dự đoán tham số tỷ lệ cục bộ:
  $$\sigma_0(\mathbf{z}) = \sigma_{\text{base}} \cdot \left(1.0 + 0.5 \tanh(\mathbf{w}^T \mathbf{z} + b)\right)$$
  $$\gamma(s, \mathbf{z}) = \frac{s^2}{s^2 + \sigma_0(\mathbf{z})^2}$$
  $$\mathcal{L}_{\text{DU-HWIoU}} = \left(1 - \mathcal{S}_{\text{DU-HWIoU}}(A, B)\right) + \lambda_{\text{unc}} \left(\log \frac{\sigma_0(\mathbf{z})}{\sigma_{\text{base}}}\right)^2$$
* **Ưu điểm**: Cho phép mô hình tự động "uốn cong" hàm Homotopy mềm hơn ở những vùng ảnh có độ nhiễu cao hoặc sương mù viễn thám.

---

### Hướng 2: Spectral Wavelet-Enhanced Homotopy (SW-HWIoU)
* **File nguồn**: [common/metrics/wavelet_h_wiou.py](file:///c:/Users/ADMIN/_Project/tiny-object-detection/common/metrics/wavelet_h_wiou.py)
* **Cơ sở toán học**:
  Sử dụng biến đổi Wavelet rời rạc 2 chiều (2D Haar DWT) để phân tách bản đồ đặc trưng thành 4 dải tần số:
  $$F \xrightarrow{\text{DWT}} \{LL, LH, HL, HH\}$$
  Tỷ số năng lượng phổ biên cạnh tần số cao:
  $$\rho_{\text{HF}}(F) = \frac{\|LH\|_2^2 + \|HL\|_2^2 + \|HH\|_2^2}{\|LL\|_2^2 + \epsilon}$$
  Điều biến số mũ Homotopy:
  $$\gamma_{\text{SW}}(s, \rho_{\text{HF}}) = \left(\frac{s^2}{s^2 + \sigma_0^2}\right) \cdot \left(1.0 - \alpha \cdot \text{sigmoid}(\rho_{\text{HF}})\right)$$
* **Ưu điểm**: Bảo toàn gradient vi mô từ các cạnh sắc nét của vật thể chỉ $2 \times 2$ pixel mà CNN thông thường hay làm mờ.

---

### Hướng 3: Oriented 2D Gaussian Homotopy (O-HWIoU cho AI-TOD-R)
* **File nguồn**: [common/metrics/oriented_h_wiou.py](file:///c:/Users/ADMIN/_Project/tiny-object-detection/common/metrics/oriented_h_wiou.py)
* **Cơ sở toán học**:
  Ánh xạ bounding box xoay 5 tham số $(x, y, w, h, \theta)$ sang phân phối chuẩn 2 chiều $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$:
  $$\boldsymbol{\Sigma} = \mathbf{R}(\theta) \begin{bmatrix} \frac{w^2}{4} & 0 \\ 0 & \frac{h^2}{4} \end{bmatrix} \mathbf{R}(\theta)^T$$
  Khoảng cách 2-Wasserstein giữa hai ellipse xoay:
  $$\mathcal{D}_{\mathcal{W}_2}^2 = \|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\|_2^2 + \text{Tr}\left(\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2 - 2 \left(\boldsymbol{\Sigma}_1^{1/2} \boldsymbol{\Sigma}_2 \boldsymbol{\Sigma}_1^{1/2}\right)^{1/2}\right)$$
  Độ tương đồng Oriented Homotopy:
  $$\mathcal{S}_{\text{O-HWIoU}}(A, B) = [\text{RotatedIoU}(A, B)]^{\gamma(s)} \cdot \exp\left( - (1 - \gamma(s)) \frac{\mathcal{D}_{\mathcal{W}_2}^2}{\sigma_0^2} \right)$$
* **Ưu điểm**: Mở rộng trực tiếp sang tập dữ liệu viễn thám có góc xoay AI-TOD-R và DOTA-v2.

---

## 3. Báo Cáo Kiểm Thử Đơn Vị (Unit Testing Certification)

* Bộ kiểm thử [paper_a/tests/test_hwiou_extensions.py](file:///c:/Users/ADMIN/_Project/tiny-object-detection/paper_a/tests/test_hwiou_extensions.py) đã thực thi và vượt qua $100\%$ các bài test:
  * `test_dynamic_uncertainty_module`: Kiểm tra shape, gradient và khoảng chặn $\sigma_0 \ge 1.0$.
  * `test_spectral_wavelet_module`: Kiểm tra phân rã Haar DWT $2\text{D}$, tỷ số $\rho_{\text{HF}} \ge 0$, và loss tiệm cận 0 khi trùng khớp hoàn hảo.
  * `test_oriented_homotopy_module`: Kiểm tra độ tương đồng Oriented IoU khi trùng góc ($\mathcal{S} \to 1.0$) và khi xoay $90^\circ$ ($\mathcal{S} < 1.0$).
* **Tổng số bài kiểm thử toàn hệ thống**: **`83/83 PASSED`** trong $1.38\text{s}$.
