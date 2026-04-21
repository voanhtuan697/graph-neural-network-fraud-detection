# Phát hiện Giao dịch Gian lận dựa trên Mạng Neural Đồ thị (GNN)
## IEEE-CIS Fraud Detection with Graph Neural Networks

### Mô tả
Đây là project khóa luận tốt nghiệp về "Phát hiện giao dịch gian lận dựa trên mạng Neural đồ thị (GNN)" sử dụng bộ dữ liệu IEEE-CIS Fraud Detection.

Project được xây dựng module hóa, chuyên nghiệp với cấu trúc rõ ràng, dễ mở rộng và tái sử dụng.

### Cấu trúc Project

```
gian_lan_ieee/
├── data/                           # Dữ liệu IEEE-CIS
├── src/                            # Source code Python modules
│   ├── config.py                   # Cấu hình tập trung
│   ├── data_loader.py              # Load & merge data
│   ├── eda.py                      # Phân tích khám phá dữ liệu
│   ├── feature_engineering.py      # Xử lý features
│   ├── graph_builder.py            # Xây dựng đồ thị heterogeneous
│   ├── graph_visualizer.py         # Trực quan đồ thị
│   ├── sampling.py                 # Xử lý mất cân bằng dữ liệu
│   ├── models/                     # Kiến trúc GNN
│   │   ├── hetero_sage.py          # HeteroGraphSAGE
│   │   ├── hetero_gat.py           # HeteroGAT
│   │   └── rgcn.py                 # RGCN
│   ├── trainer.py                  # Training loop
│   ├── evaluator.py                # Đánh giá metrics
│   └── utils.py                    # Hàm tiện ích
├── notebooks/                      # Jupyter notebooks
│   ├── 01_eda.ipynb                # EDA dataset
│   ├── 02_graph_construction.ipynb # Xây dựng đồ thị
│   ├── 03_graph_visualization.ipynb# Trực quan đồ thị
│   ├── 04_training.ipynb           # Training & experiments
│   └── 05_evaluation.ipynb         # Đánh giá & so sánh
├── output/                         # Kết quả output
│   ├── eda/                        # EDA plots
│   ├── graphs/                     # Graph visualizations
│   └── metrics/                    # Evaluation tables & charts
├── models/                         # Trained model checkpoints
├── requirements.txt                # Dependencies
└── README.md                       # File này
```

### Tính năng chính
- **EDA**: Phân tích phân phối lớp, giao dịch, giá trị thiếu, tương quan
- **Xây dựng đồ thị**: HeteroData (PyG) với 15 loại node, 29+ loại edge
- **Trực quan đồ thị**: Schema, thống kê, subgraph sampling
- **3 kiến trúc GNN**: HeteroGraphSAGE, HeteroGAT, HeteroRGCN
- **Xử lý mất cân bằng**: Focal Loss, Class Weighting, NeighborLoader
- **Đánh giá**: Precision, Recall, F1, AUC-ROC, AUC-PR, Confusion Matrix
- **Time-based split**: Tránh data leakage

### Cách sử dụng

```python
from src.config import set_seed
from src.data_loader import IEEECISDataLoader
from src.graph_builder import HeteroGraphBuilder
from src.sampling import ImbalanceSampler
from src.models import HeteroGraphSAGE
from src.trainer import GNNTrainer
from src.evaluator import ModelEvaluator

# 1. Set seed
set_seed(42)

# 2. Load data
loader = IEEECISDataLoader()
df = loader.load()

# 3. Build graph
builder = HeteroGraphBuilder(df)
data = builder.build()

# 4. Create data loaders
sampler = ImbalanceSampler(data)
loaders = sampler.get_all_loaders()
loss_fn = sampler.get_focal_loss()

# 5. Create model
model = HeteroGraphSAGE(
    metadata=data.metadata(),
    in_channels=data["txn"].x.shape[1],
)

# 6. Train
trainer = GNNTrainer(model, loss_fn, model_name="HeteroSAGE")
history = trainer.train(loaders['train'], loaders['val'])

# 7. Evaluate
y_true, y_pred, y_prob = trainer.predict(loaders['test'])
evaluator = ModelEvaluator()
evaluator.evaluate(y_true, y_pred, y_prob, model_name="HeteroSAGE")
```

### Yêu cầu hệ thống
- Python >= 3.9
- CUDA-capable GPU (khuyến nghị)
- RAM >= 16GB
- PyTorch >= 2.0
- PyTorch Geometric >= 2.4

### Cài đặt
```bash
pip install -r requirements.txt
```

**Lưu ý**: Cần cài đặt PyTorch và PyTorch Geometric phù hợp với phiên bản CUDA:
```bash
pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```
