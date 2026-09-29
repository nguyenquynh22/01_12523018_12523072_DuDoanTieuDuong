# Sổ tay ôn tập: dự đoán nguy cơ tiểu đường bằng Machine Learning

Tài liệu này tóm tắt đúng pipeline hiện tại để hai thành viên thống nhất cách trình bày và ôn vấn đáp. Đây là project học tập về phân loại dữ liệu, không phải công cụ chẩn đoán hay tư vấn y tế.

## 1. Bài toán và dữ liệu

Đây là bài toán phân loại nhị phân có giám sát: model học từ các hồ sơ đã có nhãn rồi dự đoán lớp 0/1 cho hồ sơ mới. Tệp ai-models/data/diabetes.csv hiện có 768 mẫu, 8 đặc trưng và cột Outcome. Phân bố nhãn là 500 mẫu lớp 0 (65.1%) và 268 mẫu lớp 1 (34.9%).

Các đặc trưng gồm Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction và Age. EDA trong project nên giúp trả lời: dữ liệu có bao nhiêu mẫu/lớp; giá trị nào thiếu hoặc đáng nghi; phân phối, thang đo và outlier ra sao; biến nào có liên hệ; hai lớp có chồng lấn không.

### EDA: điều cần nói đúng

- Tỷ lệ lớp 65/35 là lệch vừa phải. Nếu luôn đoán lớp đông hơn, Accuracy đã khoảng 65%, nhưng không phát hiện được mẫu lớp dương. Vì vậy không chỉ nhìn Accuracy.
- Project xem giá trị 0 ở Glucose (5 mẫu), BloodPressure (35), SkinThickness (227), Insulin (374), BMI (11) là thiếu/không hợp lệ rồi điền median. Đây là giả định xử lý của bài dựa trên ngữ nghĩa thuộc tính; cần kiểm chứng với tài liệu nguồn, không khẳng định mọi số 0 đều sai.
- Median ít bị outlier kéo lệch hơn mean. Insulin lệch phải, giá trị lớn nhất trong file là 846.
- Glucose có tương quan tuyến tính lớn nhất với Outcome trong file hiện tại (khoảng 0.467), tiếp theo BMI (0.293). Tương quan mô tả liên hệ trong mẫu, không chứng minh nhân quả và không mô tả hết tương tác phi tuyến.
- Biểu đồ phân phối, boxplot và scatter plot giúp nhìn độ lệch, outlier và vùng chồng lấn giữa lớp. EDA không thay thế đánh giá trên dữ liệu giữ riêng.

## 2. Pipeline đang chạy

1. Tách X gồm 8 đặc trưng và y là Outcome.
2. Chia phân tầng 80% train / 20% test, random_state=42: hiện 614 train, 154 test.
3. Trong pipeline, xem 0 ở 5 cột trên là thiếu, điền median và chuẩn hóa. Các bước được fit bên trong phần train của từng fold để tránh dùng thống kê từ test.
4. So sánh 4 model bằng 5-fold stratified CV trên 80% train, lấy xác suất out-of-fold.
5. Cho từng model, chọn threshold đạt Recall dương ít nhất 80%; trong các threshold khả thi, tối đa Specificity. Weighted F1 là chỉ số bổ sung/tie-break.
6. Chọn model có Specificity CV cao nhất sau điều kiện Recall. Sau đó fit từng model trên toàn bộ 80% train và đánh giá một lần trên 20% test.
7. Ứng dụng trả dự đoán của cả 4 model; nhãn “được chọn theo CV” là lựa chọn toàn cục từ dữ liệu huấn luyện, không thay đổi theo từng người nhập.

Threshold là cutoff biến xác suất thành nhãn: xác suất lớp 1 từ threshold trở lên thì gán lớp 1. Threshold đang lưu riêng cho từng model: Logistic 28.1%, SVM 25.9%, Naive Bayes 17.7%, Random Forest 32.0%. Threshold dưới 50% là chủ ý để đạt Recall mục tiêu; 0.5 chỉ là mặc định thường gặp, không mặc nhiên tối ưu.

## 3. Vì sao so sánh 4 model?

Bốn model đại diện cho cách học khác nhau, không phải tuyên bố rằng chỉ chúng mới dùng được.

| Model | Ý tưởng và lý do đưa vào | Điểm cần lưu ý |
|---|---|---|
| Logistic Regression | Baseline tuyến tính, tương đối dễ giải thích, hỗ trợ xác suất | Ranh giới cơ bản tuyến tính; nhạy với scale; xác suất vẫn cần kiểm tra calibration |
| SVM (RBF ở cấu hình này) | Tìm ranh giới có biên lớn; kernel mô tả ranh giới phi tuyến | Cần chuẩn hóa; khó giải thích; probability=True thêm chi phí và xác suất không mặc nhiên là risk đã hiệu chỉnh |
| Gaussian Naive Bayes | Dùng định lý Bayes với giả định độc lập có điều kiện và phân phối Gaussian | Nhanh, đơn giản; giả định có thể không hợp khi đặc trưng tương quan hoặc lệch |
| Random Forest | Tập hợp nhiều cây để học quan hệ phi tuyến/tương tác | Ít cần scaling; khó giải thích hơn; xác suất cần kiểm tra calibration |

Nhóm này cho phép so sánh baseline tuyến tính, kernel/margin, Bayes đơn giản và ensemble cây. Pipeline hiện chuẩn hóa cả bốn để code thống nhất; cây thường không cần StandardScaler, nhưng scaling không phải lợi thế cốt lõi của Random Forest.

## 4. Vì sao chọn “đạt Recall tối thiểu rồi tối đa Specificity”?

Recall lớp dương = TP/(TP+FN): trong số ca dương thật, tỷ lệ model phát hiện được. Specificity = TN/(TN+FP): trong số ca âm thật, tỷ lệ model nhận đúng. Tăng Recall thường làm giảm bỏ sót (FN), nhưng có thể làm tăng cảnh báo nhầm (FP); Specificity cao giúp giảm FP.

Quy tắc hiện tại là: (1) đặt mục tiêu Recall dương ít nhất 80%; (2) với mỗi model chọn threshold đạt mục tiêu; (3) trong các threshold đạt điều kiện chọn Specificity cao nhất; (4) chọn model có Specificity CV cao nhất. Lý do là bài tập mô phỏng sàng lọc, nơi nhóm chủ động đặt ưu tiên phát hiện lớp dương trước rồi giảm cảnh báo nhầm trong số phương án đạt ưu tiên đó.

Mức 80% là giả định minh họa, chưa được xác nhận bởi chuyên gia hoặc hướng dẫn lâm sàng. Không được trình bày đây là ngưỡng chuẩn y khoa. Một quy tắc có thể hợp với mục tiêu bài tập nhưng không đồng nghĩa model đã hữu ích hoặc an toàn trong thực tế.

Các lựa chọn khác:

- Threshold 0.5 rồi so Weighted F1: đơn giản, tái lập và khớp bảng Colab cũ; không đảm bảo Recall đạt mức tối thiểu.
- Tối đa Youden J = Recall + Specificity - 1: chọn cân bằng hai tỷ lệ; ngầm xem hai loại lỗi có trọng số ngang nhau.
- Tối đa Recall: hạn chế bỏ sót, nhưng có thể tạo rất nhiều cảnh báo nhầm; không tự giới hạn false positive.
- ROC-AUC/PR-AUC: đánh giá thứ hạng trên nhiều threshold; không chọn ra cutoff triển khai.
- Decision curve/net benefit: cân nhắc lợi ích và tác hại theo hành động sau dự đoán; project chưa có dữ liệu về hành động/hậu quả để áp dụng có căn cứ.

## 5. Đọc kết quả hiện tại

| Model | Threshold | CV Recall + | CV Specificity | CV Weighted F1 | Test Recall + | Test Specificity | Test Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 28.1% | 80.4% | 69.0% | 73.6% | 83.3% | 67.0% | 73.3% |
| SVM | 25.9% | 80.4% | 70.3% | 74.4% | 75.9% | 65.0% | 69.5% |
| Naive Bayes | 17.7% | 80.4% | 67.0% | 72.3% | 77.8% | 63.0% | 68.9% |
| Random Forest | 32.0% | 80.4% | 69.8% | 74.1% | 79.6% | 67.0% | 72.1% |

SVM được chọn theo Specificity CV cao nhất (70.3%), nhưng chỉ hơn Random Forest 0.5 điểm phần trăm. Trên tập test hiện tại, Random Forest tốt hơn ở Recall, Specificity và Weighted F1 trong các chỉ số trên. Cách nói trung thực: SVM đứng đầu theo quy tắc CV đã định; kết quả test cho thấy ưu thế đó không ổn định ở lần chia này, cần đánh giá thêm. Không nên nói SVM vượt trội rõ rệt.

Lưu ý phương pháp: code lấy dự đoán OOF rồi dùng cùng các nhãn OOF để chọn threshold và tính metric CV. Vì dữ liệu OOF tham gia cả chọn lẫn chấm threshold, kết quả CV có thể lạc quan. Nested cross-validation (vòng trong chọn threshold, vòng ngoài đánh giá) hoặc validation độc lập sẽ chặt chẽ hơn. Test 154 mẫu giúp kiểm tra trên dữ liệu chưa dùng trong việc fit model, nhưng một lần chia nhỏ vẫn có bất định.

## 6. Các chỉ số phải phân biệt

| Chỉ số | Ý nghĩa |
|---|---|
| Accuracy = (TP+TN)/tổng mẫu | Tỷ lệ dự đoán đúng tổng thể; có thể bị lớp đông chi phối |
| Precision dương = TP/(TP+FP) | Trong các cảnh báo dương, bao nhiêu là dương thật |
| Positive Recall/Sensitivity = TP/(TP+FN) | Trong các ca dương thật, model phát hiện được bao nhiêu |
| Specificity = TN/(TN+FP) | Trong các ca âm thật, model nhận đúng bao nhiêu |
| F1 | Trung bình điều hòa Precision và Recall của một lớp |
| Weighted F1 | F1 hai lớp lấy trung bình theo số lượng mẫu từng lớp; nên xem kèm Recall dương |
| ROC-AUC | Khả năng xếp mẫu dương cao hơn mẫu âm trên nhiều cutoff |
| PR-AUC | Quan hệ Precision-Recall trên nhiều cutoff; hữu ích khi quan tâm lớp dương |
| Threshold | Điểm cắt biến score/xác suất thành nhãn |

Confusion matrix: TP là dương đúng; FN là bỏ sót dương; TN là âm đúng; FP là cảnh báo dương nhầm. Trong phân loại một nhãn, weighted recall bằng accuracy về mặt tổng hợp; không nên xem hai số đó là hai bằng chứng độc lập.

## 7. Câu hỏi giảng viên có thể hỏi

**Tại sao chọn 4 model này?**  
Chúng đại diện các giả định/cách học khác nhau: tuyến tính, kernel, Bayes và ensemble cây. Đây là so sánh có chủ đích trong phạm vi project, không phải khẳng định đã thử mọi thuật toán.

**Tại sao cần EDA?**  
Để hiểu phân bố, chất lượng dữ liệu, nhãn, giá trị 0 đáng nghi, outlier và quan hệ giữa biến; từ đó thiết kế tiền xử lý và nhận biết giới hạn.

**Tại sao thay một số 0 bằng median?**  
Code coi 0 tại năm thuộc tính sinh học là missing/invalid; median bền với lệch/outlier hơn mean. Đây là giả định cần ghi rõ và kiểm tra với nguồn dữ liệu.

**Tại sao 80/20 và stratify?**  
Giữ tập test riêng để đánh giá cuối và bảo toàn gần tỷ lệ nhãn ở cả hai phần. 80/20 là lựa chọn thực dụng cho bài này, không phải quy tắc duy nhất đúng.

**Cross-validation có tác dụng gì?**  
Dùng nhiều fold trong train để so model/chọn threshold ít phụ thuộc vào một lần chia hơn. Test vẫn giữ riêng để đánh giá cuối.

**Recall càng cao càng tốt không?**  
Không nếu xét riêng lẻ. Recall cao thường giảm FN nhưng có thể tăng FP. Cần xét Specificity và chi phí của từng lỗi.

**Tại sao SVM được khuyên dùng dù RF tốt hơn trên test?**  
Quy tắc đã chọn model bằng Specificity CV sau điều kiện Recall; SVM là 70.3%, RF 69.8%. Trên test này RF tốt hơn ở các chỉ số hiển thị. Vì vậy SVM chỉ là model được chọn theo quy tắc CV, chưa chứng minh tổng quát tốt hơn.

**80% Recall lấy từ đâu?**  
Đây là giả định mô phỏng cho bài tập, không phải mục tiêu lâm sàng đã được xác nhận. Thực tế cần ý kiến chuyên gia, bối cảnh sử dụng và phân tích lợi ích-tác hại.

**Xác suất model trả về có phải xác suất bệnh thật không?**  
Không thể mặc định như vậy. Cần đánh giá calibration trên quần thể phù hợp; hiện tại đây là xác suất/score do model ước lượng trên dữ liệu học.

**Glucose tương quan cao nhất, sao không chỉ dùng Glucose?**  
Tương quan đơn biến không thể hiện toàn bộ đóng góp kết hợp/tương tác. Cần so sánh mô hình/feature bằng validation, không chỉ dựa vào một hệ số EDA.

**Giới hạn lớn nhất của project?**  
Dữ liệu nhỏ; giả định về số 0; 80% Recall chưa có căn cứ lâm sàng; chưa có external validation/calibration/net-benefit; kết quả CV threshold có thể lạc quan; test hiện nhỏ và RF thắng SVM trên một số chỉ số.

## 8. Trả lời tổng kết trong 30 giây

“Nhóm thực hiện phân loại nhị phân trên 8 thuộc tính và so sánh Logistic Regression, SVM, Gaussian Naive Bayes và Random Forest. Dữ liệu chia phân tầng 80/20; trong train dùng 5-fold CV. Để mô phỏng sàng lọc, nhóm đặt Recall dương tối thiểu 80% rồi tối đa Specificity. Đây là giả định học thuật, không phải chuẩn y khoa. SVM được chọn vì Specificity CV 70.3%, cao nhất, nhưng chỉ hơn RF 0.5 điểm phần trăm; trên test lần này RF tốt hơn ở các chỉ số được báo cáo. Vì vậy chưa thể kết luận SVM vượt trội và cần đánh giá thêm bằng nested CV hoặc dữ liệu ngoài.”

## 9. Nguồn để học thêm

- Scikit-learn: [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), [GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html), [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html), [model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html).
- Riley RD et al., [Developing clinical prediction models: a step-by-step guide](https://www.bmj.com/content/386/bmj-2023-078276), BMJ, 2024.
- Riley RD et al., [How to undertake an external validation study](https://www.bmj.com/content/384/bmj-2023-074820), BMJ, 2024.
- Parikh R et al., [Understanding and using sensitivity, specificity and predictive values](https://pmc.ncbi.nlm.nih.gov/articles/PMC2636062/), Indian Journal of Ophthalmology, 2008.

## 10. File project nên mở khi cần kiểm chứng

- ai-models/src/train.py: pipeline, CV, threshold selection, artifacts.
- ai-models/colab/03_train.ipynb: notebook huấn luyện.
- ai-models/models/decision_config.json: threshold, model được chọn, chỉ số CV/test.
- ai-models/service/main.py: suy luận và áp dụng threshold.
- App/backend/app/main.py và App/frontend/src/App.tsx: API response và hiển thị.
- README.md: hướng dẫn project/API.
