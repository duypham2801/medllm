# Test Hướng Dẫn - Model 3B + Dummy Data

## ✅ Setup Hoàn Tất

Tất cả các lỗi đã được sửa:
1. ✅ Packages installed (transformers 4.57.1, peft 0.18.0, etc.)
2. ✅ Code fixed (imports, cuda calls, etc.)
3. ✅ Dummy dataset created (không cần download data)
4. ✅ Test config created (TinyLlama 1.1B)

## 🚀 Để Test Code (KHÔNG CẦN DOWNLOAD DATA)

### Option 1: Test Nhanh Với Model Nhỏ (~2GB download)

```bash
# Activate environment
conda activate llm

# Chạy training test với dummy data
bash scripts/test_training.sh
```

Sẽ:
- Auto-download TinyLlama-1.1B (~2GB) từ HuggingFace
- Tạo 10 ảnh dummy tự động
- Train 3 steps để verify code hoạt động
- Không cần COCO/RefCOCO dataset

### Option 2: Chọn Model Khác (3B)

Edit `config/training_configs/test_3b_dummy.py` line 96, uncomment model bạn muốn:

**Phi-3.5-mini (3.8B) - Better quality**
```python
model_name_or_path="microsoft/Phi-3.5-mini-instruct",  # ~8GB download
```

**StableLM-3B**
```python
model_name_or_path="stabilityai/stablelm-3b-4e1t",  # ~6GB download
```

**TinyLlama-1.1B (default) - Fastest**
```python
model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # ~2GB download
```

## 📊 Kết Quả Mong Đợi

Nếu chạy thành công, bạn sẽ thấy:

```
Step 1/3: loss=X.XXX
Step 2/3: loss=X.XXX
Step 3/3: loss=X.XXX
✓ Training test completed!
```

## 🎯 Training Thật Với Data Thật

Sau khi verify code chạy được, để training với data thật:

### 1. Download LLaVA checkpoint + RefCOCO data

```bash
bash scripts/download_data.sh
```

Sẽ download:
- LLaVA-v1.5-7B (~13GB)
- COCO images (~13-19GB)
- RefCOCO annotations (~500MB)

### 2. Training với GPU 4GB

```bash
bash scripts/run_4gb.sh
```

Dùng config: `perception_1gpu_4gb_lora.py` với:
- DeepSpeed ZeRO-3 + CPU offloading
- 8-bit quantization
- LoRA rank=8
- Batch size=1

## 📝 Files Đã Tạo

```
✓ config/training_configs/test_3b_dummy.py  - Test config
✓ mllm/dataset/dummy_dataset.py            - Dummy dataset
✓ scripts/test_training.sh                 - Test script
✓ scripts/note.txt                         - Model options
✓ TEST_README.md                           - This file
```

## 🐛 Troubleshooting

### ImportError with peft
```bash
pip install --upgrade 'transformers>=4.45.0'
```

### CUDA Out of Memory
- Dùng TinyLlama thay vì model lớn hơn
- Giảm `image_token_len` từ 64 xuống 32
- Giảm `max_length` từ 512 xuống 256

### Model download slow
- Dùng HuggingFace mirror: `HF_ENDPOINT=https://hf-mirror.com`
- Hoặc download manually và update `model_name_or_path`

## ℹ️ Thông Tin GPU

- **GPU**: NVIDIA GeForce GTX 1650 with Max-Q Design
- **VRAM**: 3.8 GB
- **CUDA**: 12.1
- **Limitations**:
  - Không support TF32 (Ampere+ only)
  - Training sẽ chậm với CPU offloading
  - Chỉ phù hợp LoRA, không full fine-tune

## ✨ Next Steps

1. **Test code**: `bash scripts/test_training.sh` (5-10 phút)
2. **Download data**: `bash scripts/download_data.sh` (1-2 giờ)
3. **Real training**: `bash scripts/run_4gb.sh` (3-6 giờ)

---

**Prepared by**: Claude Code
**Last Updated**: 2025-11-14
