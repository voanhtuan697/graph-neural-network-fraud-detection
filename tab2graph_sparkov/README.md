# Graph Feature Effectiveness in Fraud Detection — Sparkov Dataset

## Mô tả
Project so sánh hiệu quả của đặc trưng đồ thị (graph features) trong bài toán phát hiện gian lận giao dịch thẻ tín dụng sử dụng dataset Sparkov.

## Phương pháp

### Giai đoạn 1: Tabular Features Only
Huấn luyện 4 mô hình ML trên đặc trưng bảng gốc (temporal, geographic, demographic, financial).

### Giai đoạn 2: Graph Features
Chuyển dữ liệu sang đồ thị không đồng nhất (Heterogeneous Graph) với:
- **Node types**: Customer, Merchant, Category
- **Edge types**: Customer↔Merchant (TRANSACTS_AT), Merchant↔Category (BELONGS_TO)

Trích xuất 12 nhóm đặc trưng đồ thị:
1. Số merchant duy nhất của user
2. Tần suất giao dịch
3. Trung bình/độ lệch chuẩn số tiền
4. Tỷ lệ gian lận của merchant
5. Bậc (degree) của merchant
6. Degree centrality
7. PageRank
8. Community detection (Louvain)
9. Neighbor fraud ratio
10. Shared fraud entity count
11. Time-based burst
12. Sequence pattern

### So sánh 3 kịch bản
1. **Tabular Only**: Chỉ dùng đặc trưng bảng
2. **Graph Only**: Chỉ dùng đặc trưng đồ thị
3. **Tabular + Graph**: Kết hợp cả hai

## Mô hình
- Logistic Regression (`class_weight='balanced'`)
- Random Forest (`class_weight='balanced'`)
- XGBoost (`scale_pos_weight`, early stopping)
- LightGBM (`is_unbalance=True`)

## Độ đo
Precision, Recall, F1-Score (cho lớp fraud = 1)

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chuẩn bị dữ liệu

Đặt 2 file CSV vào thư mục `data/`:
- `data/SparkovTrain.csv`
- `data/SparkovTest.csv`

Download từ: https://www.kaggle.com/datasets/kartik2112/fraud-detection

## Chạy pipeline

```bash
python notebooks/main_pipeline.py
```

## Output
Kết quả được lưu trong `results/`:
- `comparison_results.csv` — Bảng so sánh chi tiết
- `comparison_bars.png` — Biểu đồ so sánh Precision/Recall/F1
- `improvement_bars.png` — Biểu đồ cải thiện
- `feature_importance_combined.png` — Feature importance (Combined)
- `confusion_matrices.png` — Confusion matrices
- `graph_feature_analysis.png` — Phân tích sức mạnh graph features

## Cấu trúc project

```
tabular_graph_Sparkov/
├── data/                          # Sparkov CSV files
├── results/                       # Output (metrics, plots)
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Load, preprocess, downsample
│   ├── graph_builder.py           # Build heterogeneous graph
│   ├── graph_features.py          # Extract 12 graph feature groups
│   ├── model_trainer.py           # Train & evaluate 4 ML models
│   └── visualization.py           # Charts, comparison tables
├── notebooks/
│   └── main_pipeline.py           # Main executable pipeline
├── requirements.txt
└── README.md
```
