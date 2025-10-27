# Transformer From Scratch

## 环境
推荐 Python 3.10+, PyTorch 2.x

安装依赖:
```bash
pip install -r requirements.txt
```
## 硬件要求
GPU: NVIDIA GPU with ≥ 8GB VRAM (推荐)\
内存: ≥ 16GB RAM\
存储: ≥ 2GB 可用空间
## 运行
```bash
bash scripts/run.sh
```
或者
```bash
python src/train.py --config base.yaml --seed 42 --device cpu
```
结果会保存到 results/，包括训练曲线 training_curve.png、模型权重 best_model.pt




