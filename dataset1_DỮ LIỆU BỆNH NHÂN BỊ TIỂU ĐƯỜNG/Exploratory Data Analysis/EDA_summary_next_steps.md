# Tổng hợp EDA và hướng xử lý tiếp theo

File này tổng hợp nội dung các notebook EDA trong thư mục `Exploratory Data Analysis`, bao gồm: sau bước EDA dataset có thêm cột nào, cần lưu ý vấn đề gì, và nên làm gì tiếp theo để chuẩn bị cho các bài toán phân loại, phân cụm và luật kết hợp.

---

## 1. Các file EDA đã có

### File 1: `01_overview_data_quality.ipynb`

**Mục tiêu:** Tổng quan dữ liệu và đánh giá chất lượng dữ liệu.

**Nội dung chính:**

- Đọc file `diabetic_data.csv`, xử lý dấu `?` thành missing value (`NaN`).
- Kiểm tra kích thước dữ liệu: số dòng, số cột.
- Xem `head`, `info`, `describe`.
- Chia nhóm cột: ID, nhân khẩu học, nhập viện, xét nghiệm/chẩn đoán, thuốc, nhãn.
- Kiểm tra missing value theo số lượng và tỷ lệ phần trăm.
- Vẽ biểu đồ missing value.
- Kiểm tra duplicate:
  - Dòng trùng hoàn toàn.
  - `encounter_id` có bị lặp hay không.
  - `patient_nbr` có bị lặp để biết một bệnh nhân có nhiều lần nhập viện hay không.
- So sánh số lượt nhập viện với số bệnh nhân duy nhất.
- Kiểm tra unique value để tìm các cột gần như vô dụng.

**Kết luận quan trọng:**

- `encounter_id`: cột định danh từng lần nhập viện, không nên dùng làm feature cho mô hình.
- `patient_nbr`: cột định danh bệnh nhân, có thể dùng để phân tích tần suất nhập viện, nhưng không nên đưa trực tiếp vào mô hình.
- `weight`: missing rất cao, nên cân nhắc bỏ.
- `payer_code`, `medical_specialty`: missing nhiều, có thể thử hai hướng: bỏ cột hoặc điền `Unknown`.
- `examide`, `citoglipton`: nếu chỉ có một giá trị duy nhất thì nên bỏ.

---

### File 2: `02_target_demographic_admission_eda.ipynb`

**Mục tiêu:** Phân tích biến mục tiêu, nhân khẩu học và thông tin nhập viện.

**Nội dung chính:**

- Đọc dữ liệu và xử lý `?` thành missing value.
- Tạo cột `readmitted_binary`:
  - `0`: `readmitted = NO`
  - `1`: `readmitted = <30` hoặc `>30`
- Phân tích phân bố `readmitted`: `NO`, `>30`, `<30`.
- EDA theo:
  - `age`
  - `gender`
  - `race`
  - `time_in_hospital`
  - `admission_type_id`
  - `admission_source_id`
  - `discharge_disposition_id`
- Tạo mapping để diễn giải các cột ID nhập viện/xuất viện:
  - `admission_type`
  - `admission_source`
  - `discharge_disposition`
- Vẽ các biểu đồ:
  - Bar chart phân bố `readmitted`.
  - Bar chart phân bố `age`.
  - Stacked bar `readmitted` theo `age`.
  - Bar chart `gender`.
  - Stacked bar `readmitted` theo `gender`.
  - Histogram `time_in_hospital`.
  - Boxplot `time_in_hospital` theo `readmitted`.
  - Bar chart `admission_type_id`.

**Kết luận quan trọng:**

- `readmitted` có thể dùng dạng 3 lớp hoặc chuyển thành binary tùy bài toán.
- Nhóm tuổi cao cần được chú ý khi xem xu hướng tái nhập viện.
- `time_in_hospital` là biến quan trọng, có thể phản ánh mức độ nặng/phức tạp của ca bệnh.
- `admission_type_id`, `admission_source_id`, `discharge_disposition_id` nên giữ lại nhưng cần mapping/encoding trước khi đưa vào mô hình.
- `race` cần xử lý missing, nên tạo nhóm `Unknown`.

---

### File 3: `03_medical_drug_diagnosis_eda.ipynb`

**Mục tiêu:** Phân tích các biến y tế chuyên sâu hơn để phục vụ phân loại, phân cụm và luật kết hợp.

**Nội dung chính:**

- Đọc dữ liệu và tạo `readmitted_binary`.
- Phân tích các biến xét nghiệm/điều trị:
  - `num_lab_procedures`
  - `num_procedures`
  - `num_medications`
  - `number_diagnoses`
- So sánh trung bình các biến y tế theo `readmitted`.
- Phân tích tiền sử y tế:
  - `number_outpatient`
  - `number_emergency`
  - `number_inpatient`
- Phân tích A1C và glucose:
  - `A1Cresult`
  - `max_glu_serum`
- Phân tích thuốc phổ biến:
  - `insulin`
  - `metformin`
  - `glipizide`
  - `glyburide`
  - `pioglitazone`
- Phân tích thay đổi thuốc:
  - `change`
  - `diabetesMed`
- Phân tích chẩn đoán:
  - `diag_1`
  - `diag_2`
  - `diag_3`
- Gom nhóm ICD-9 thành:
  - `diag_1_group`
  - `diag_2_group`
  - `diag_3_group`
- Kiểm tra outlier bằng boxplot.
- Vẽ correlation matrix giữa các biến số và `readmitted_binary`.

**Kết luận quan trọng:**

- Các biến `num_medications`, `num_lab_procedures`, `number_diagnoses`, `number_inpatient` nên được giữ để thử trong mô hình.
- Các biến tiền sử y tế như `number_inpatient`, `number_emergency`, `number_outpatient` không nên xóa outlier tự động vì có thể là tín hiệu quan trọng.
- Các biến thuốc phù hợp cho cả phân loại và luật kết hợp.
- Nên dùng `diag_1_group`, `diag_2_group`, `diag_3_group` thay vì chỉ dùng mã chẩn đoán gốc vì ICD-9 có quá nhiều giá trị.

---

## 2. Sau EDA dữ liệu có thêm gì?

Trong các notebook EDA, dữ liệu gốc được đọc lại từ `diabetic_data.csv`. Các cột mới được tạo trong quá trình phân tích gồm:

- `readmitted_binary`: biến mục tiêu nhị phân.
- `admission_type`: diễn giải từ `admission_type_id`.
- `admission_source`: diễn giải từ `admission_source_id`.
- `discharge_disposition`: diễn giải từ `discharge_disposition_id`.
- `race_display`: bản hiển thị của `race`, trong đó missing được gán thành `Unknown`.
- `diag_1_group`: nhóm bệnh lớn từ `diag_1`.
- `diag_2_group`: nhóm bệnh lớn từ `diag_2`.
- `diag_3_group`: nhóm bệnh lớn từ `diag_3`.

**Lưu ý:** Các cột này hiện mới được tạo trong notebook, chưa được lưu thành file CSV clean riêng. Bước tiếp theo nên tạo file clean/preprocessed riêng để các phần sau dùng thống nhất.

---

## 3. Các cột nên bỏ hoặc không dùng trực tiếp

### Nên bỏ khỏi feature mô hình

- `encounter_id`: ID của lần nhập viện.
- `patient_nbr`: ID bệnh nhân.
- `examide`: nếu chỉ có một giá trị.
- `citoglipton`: nếu chỉ có một giá trị.
- `weight`: missing quá cao, nên bỏ nếu không có chiến lược xử lý riêng.

### Cần cân nhắc

- `payer_code`: missing nhiều, có thể bỏ hoặc điền `Unknown`.
- `medical_specialty`: missing nhiều, có thể bỏ hoặc điền `Unknown`.
- `diag_1`, `diag_2`, `diag_3`: quá nhiều giá trị, nên gom nhóm ICD-9 hoặc chỉ giữ top mã phổ biến.

---

## 4. Các cột nên giữ cho mô hình phân loại

### Nhóm nhân khẩu học

- `race` hoặc `race_display`
- `gender`
- `age`

### Nhóm nhập viện

- `admission_type_id`
- `admission_source_id`
- `discharge_disposition_id`
- `time_in_hospital`

### Nhóm xét nghiệm/điều trị

- `num_lab_procedures`
- `num_procedures`
- `num_medications`
- `number_diagnoses`

### Nhóm tiền sử y tế

- `number_outpatient`
- `number_emergency`
- `number_inpatient`

### Nhóm xét nghiệm A1C/glucose

- `A1Cresult`
- `max_glu_serum`

### Nhóm thuốc

- `metformin`
- `glipizide`
- `glyburide`
- `pioglitazone`
- `insulin`
- `change`
- `diabetesMed`

### Nhóm chẩn đoán

- `diag_1_group`
- `diag_2_group`
- `diag_3_group`

### Biến mục tiêu

- Đa lớp: `readmitted`
- Nhị phân: `readmitted_binary`

---

## 5. Sau bước EDA nên làm gì?

### Bước 1: Tạo notebook/file clean dữ liệu

Nên tạo file tiếp theo:

```text
04_data_cleaning_feature_engineering.ipynb