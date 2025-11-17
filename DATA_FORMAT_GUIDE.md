# Medical Data Format Guide for MedGemma

## Tổng quan

MedGemma hỗ trợ 7 loại medical imaging modalities với nhiều task khác nhau:
- **CT (Computed Tomography)**: Phát hiện khối u, segmentation cơ quan
- **MRI (Magnetic Resonance Imaging)**: Phát hiện thương tổn não, phân loại mô
- **X-ray**: Phát hiện bất thường, phân loại bệnh
- **Ultrasound**: Theo dõi thai nhi, đánh giá cơ quan
- **Fundus Photography**: Bệnh đái tháođáo, glaucoma
- **Pathology**: Đếm tế bào, phát hiện ung thư
- **Endoscopy**: Phát hiện polyp, phân loại thương tổn

## 1. Định dạng JSONL cơ bản

Mỗi dòng là một JSON object với cấu trúc:

```json
{
  "image": "path/to/image.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "Đây là ảnh X-ray phổi có gì bất thường không?"
    },
    {
      "from": "gpt",
      "value": "Ảnh X-ray cho thấy có dấu hiệu viêm phổi ở lưới phổi phải..."
    }
  ],
  "modality": "xray",
  "dataset": "vinchest_xray"
}
```

### Các trường bắt buộc:
- `image`: Đường dẫn đến file ảnh
- `conversations`: Mảng các cuộc hội thoại (format LLaVA style)
- `modality`: Loại imaging modalities

### Các trường tùy chọn:
- `dataset`: Tên dataset gốc
- `task`: Loại task (classification, detection, segmentation, vqa, report)

## 2. Format theo Task

### 2.1 Classification (Phân loại)

```json
{
  "image": "data/chest_xray/patient001_view1_frontal.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "Hãy phân loại ảnh X-ray ngực này: Normal, Pneumonia, hay COVID-19?"
    },
    {
      "from": "gpt",
      "value": "Kết quả: PNEUMONIA\n\nĐặc điểm: Có hình ảnh mờ đục ở lưới phổi bên phải, góc mũi tim bị mờ..."
    }
  ],
  "modality": "xray",
  "dataset": "chest_xray",
  "task": "classification",
  "label": "pneumonia"
}
```

### 2.2 Detection (Phát hiện đối tượng)

```json
{
  "image": "data/ct_scan/brain_001.png",
  "conversations": [
    {
      "from": "human",
      "value": "Tìm và đánh dấu vị trí các khối u trong ảnh MRI não này."
    },
    {
      "from": "gpt",
      "value": "Tìm thấy 1 khối u:\n- Vị trí: [156, 234, 78, 112] (x, y, w, h)\n- Loại: U não ác tính\n- Đặc điểm: Vùng tăng sáng bất thường ở thùy trái não"
    }
  ],
  "modality": "mri",
  "dataset": "brain_mri",
  "task": "detection",
  "boxes": [[156, 234, 78, 112]],
  "categories": ["tumor"]
}
```

### 2.3 Segmentation (Phân vùng)

```json
{
  "image": "data/pathology/slide_001.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "Segment và đếm số lượng tế bào ung thư trong ảnh mô bệnh học này."
    },
    {
      "from": "gpt",
      "value": "Segment hoàn thành:\n- Số lượng tế bào ung thư: 27\n- Vị trí tế bào: [[mask_1], [mask_2], ...]\n- Mật độ: 15 tế bào/mm²"
    }
  ],
  "modality": "pathology",
  "dataset": "camelyon16",
  "task": "segmentation",
  "masks": ["mask_path_001.png", "mask_path_002.png", ...],
  "counts": 27
}
```

### 2.4 VQA (Visual Question Answering)

```json
{
  "image": "data/fundus/retina_001.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "Đã thấy dấu hiệu của bệnh đái tháo đường ở đáy mắt chưa?"
    },
    {
      "from": "gpt",
      "value": "Có, đã thấy các dấu hiệu của retinopathy do đái tháo đường:\n- Xuất huyết điểm: 5 điểm\n- Exudates: 3 vùng\n- Phì đại thị lực: Rõ rệt\n- Kết luận: DR Proliferative"
    }
  ],
  "modality": "fundus",
  "dataset": "messidor",
  "task": "vqa",
  "question": "Đã thấy dấu hiệu của bệnh đái tháo đường ở đáy mắt chưa?",
  "answer": "Có, retinopathy proliferative"
}
```

### 2.5 Report Generation (Tạo báo cáo)

```json
{
  "image": "data/ct_chest/ct_001.png",
  "conversations": [
    {
      "from": "human",
      "value": "Hãy viết báo cáo radiology chi tiết cho ảnh CT lồng ngực này."
    },
    {
      "from": "gpt",
      "value": "BÁO CÁO CT LỒNG NGỰC\n\nKẾT QUẢ CHẠN CẮT:\n- Tim: Kích thước bình thường, không có phì đại thất\n- Phổi: \n  * Lợi phải: Có tổn thương viêm ở thùy trên, kích thước 3x2cm\n  * Lợi trái: Bình thường\n- Trung thất: Không có hạch lympho to\n- Xương: Không có dấu hiệu di căn xương\n\nKẾT LUẬN:\n- Tổn thương viêm thùy trên phải (có thể là tuberculosis)\n- Đề nghị làm thêm xét nghiệm và chụp cắt lớp vi tính\n- Theo dõi sau 2 tuần điều trị"
    }
  ],
  "modality": "ct",
  "dataset": "luna16",
  "task": "report"
}
```

### 2.6 Multiple Tasks (Nhiều task trong 1 ảnh)

```json
{
  "image": "data/ultrasound/fetal_001.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "Phân tích ảnh siêu âm thai nhi này: đo kích thước, phát hiện bất thường, và viết báo cáo ngắn."
    },
    {
      "from": "gpt",
      "value": "PHÂN TÍCH SIÊU ÂM THAI NHI (22 tuần)\n\nĐO LƯỜNG:\n- BPD (Đường kính lưỡng đỉnh): 54mm (bình thường)\n- HC (Chu vi đầu): 200mm (bình thường)\n- AC (Chu vi bụng): 185mm (bình thường)\n- FL (Chiều dài xương đùi): 38mm (bình thường)\n\nPHÁT HIỆN:\n- Tim: Báo động bình thường, 4 buồng tim đều\n- Não: Không có ventriculomegaly\n- Cột sống: Không có spina bifida\n- Thận: 2 thận đều có kích thước bình thường\n\nKẾT LUẬN:\n- Thai phát triển bình thường, tương đương 22 tuần\n- Không phát hiện bất thường quan trọng\n- Ước tính cân nặng lúc sinh: 2800-3200g"
    }
  ],
  "modality": "ultrasound",
  "dataset": "fetal_ultrasound",
  "task": ["measurement", "detection", "report"]
}
```

## 3. Format COCO cho Detection/Segmentation

Nếu bạn có dữ liệu theo format COCO:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "ct_scan_001.jpg",
      "width": 512,
      "height": 512,
      "modality": "ct",
      "patient_id": "P001",
      "study_date": "2024-01-15"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [156, 234, 78, 112],
      "area": 8736,
      "segmentation": [[polygon_points]],
      "label": "tumor",
      "confidence": 0.95
    }
  ],
  "categories": [
    {"id": 1, "name": "tumor", "supercategory": "lesion"},
    {"id": 2, "name": "cyst", "supercategory": "lesion"}
  ]
}
```

## 4. Tập dataset mẫu

### 4.1 Vietnamese Medical VQA Dataset

```json
{"image": "data/xray/chest_001.png", "conversations": [{"from": "human", "value": "Ảnh X-ray này có dấu hiệu của bệnh COVID-19 không?"}, {"from": "gpt", "value": "Không, ảnh X-ray này bình thường, không có dấu hiệu viêm phổi do COVID-19."}], "modality": "xray"}

{"image": "data/ct/brain_001.png", "conversations": [{"from": "human", "value": "Tìm vị trí xuất huyết trong ảnh MRI não này."}, {"from": "gpt", "value": "Xuất huyết tại vị trí: [120, 180, 45, 60] ở thùy trán trái, kích thước khoảng 15ml."}], "modality": "mri", "task": "detection"}

{"image": "data/fundus/retina_001.jpg", "conversations": [{"from": "human", "value": "Chẩn đoán bệnh từ ảnh đáy mắt này."}, {"from": "gpt", "value": "Chẩn đoán: Diabetic Retinopathy Moderate\nĐặc điểm: Có microaneurysms và dot hemorrhages, nhưng chưa có neovascularization."}], "modality": "fundus", "task": "classification"}
```

### 4.2 Pathology Cell Counting Dataset

```json
{"image": "data/pathology/breast_001.tif", "conversations": [{"from": "human", "value": "Đếm số lượng tế bào ác tính trong ảnh mô bệnh học ung thư vú này."}, {"from": "gpt", "value": "Đếm được 23 tế bào ác tính.\nĐặc điểm:\n- Nuclear size: Large (>25μm)\n- Nuclear shape: Irregular\n- Chromatin: Coarse\n- Mitoses: 3 figures\nChẩn đoán: High-grade invasive ductal carcinoma"}], "modality": "pathology", "task": ["counting", "classification"], "counts": 23}
```

## 5. Cấu trúc thư mục đề xuất

```
data/
├── medical_datasets/
│   ├── viet_xray/
│   │   ├── images/
│   │   ├── viet_xray.jsonl
│   │   └── README.md
│   ├── brain_mri/
│   │   ├── t1/
│   │   ├── t2/
│   │   ├── flair/
│   │   ├── brain_mri.jsonl
│   │   └── masks/
│   ├── fundus_retina/
│   │   ├── images/
│   │   ├── fundus_retina.jsonl
│   │   └── segmentations/
│   └── pathology_cells/
│       ├── slides/
│       ├── pathology_cells.jsonl
│       └── annotations/
```

## 6. Script chuyển đổi dữ liệu

Để convert từ format khác sang JSONL:

```python
import json
from pathlib import Path

def convert_coco_to_jsonl(coco_file, output_file, modality):
    """Convert COCO format to MedGemma JSONL"""
    with open(coco_file) as f:
        coco_data = json.load(f)

    # Map image_id to image info
    images = {img['id']: img for img in coco_data['images']}

    # Group annotations by image
    image_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)

    # Convert to JSONL
    output_lines = []
    for img_id, img_info in images.items():
        anns = image_annotations.get(img_id, [])

        # Build conversation
        if anns:
            # Detection/segmentation task
            human_msg = "Tìm và phân tích tất cả các bất thường trong ảnh medical này."
            gpt_msg = f"Tìm thấy {len(anns)} bất thường:\n"

            for i, ann in enumerate(anns, 1):
                if 'bbox' in ann:
                    x, y, w, h = ann['bbox']
                    gpt_msg += f"{i}. Vị trí: [{x}, {y}, {w}, {h}]"
                    if 'segmentation' in ann:
                        gpt_msg += " (có segmentation mask)"
                    gpt_msg += "\n"
        else:
            # Classification/VQA task
            human_msg = "Phân tích và mô tả ảnh medical này."
            gpt_msg = "Ảnh này không có annotations, cần phân tích tổng quan."

        line = {
            "image": img_info['file_name'],
            "conversations": [
                {"from": "human", "value": human_msg},
                {"from": "gpt", "value": gpt_msg}
            ],
            "modality": modality,
            "image_id": img_id
        }

        output_lines.append(json.dumps(line, ensure_ascii=False))

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))

# Usage
# convert_coco_to_jsonl('annotations.json', 'output.jsonl', 'ct')
```

## 7. Tips quan trọng

### 7.1 Quality Control
- Luôn có bác sĩ review annotations
- Use inter-annotator agreement > 0.8
- Remove low-quality images

### 7.2 Data Augmentation
- Rotation: Medical images thường có hướng cố định
- Intensity: Cẩn thận với intensity normalization
- Noise: Thêm realistic medical imaging noise

### 7.3 Privacy & Ethics
- Remove all patient identifiers
- Follow HIPAA/GDPR guidelines
- Get proper consent for data usage

### 7.4 Class Balance
- Balance between normal and abnormal cases
- Consider prevalence rates in real population
- Use stratified sampling

## 8. Validation Script

```python
import json
from pathlib import Path
from PIL import Image

def validate_medgemma_dataset(jsonl_file):
    """Validate medical dataset format"""
    errors = []
    warnings = []

    with open(jsonl_file) as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)

                # Check required fields
                if 'image' not in data:
                    errors.append(f"Line {line_num}: Missing 'image' field")

                if 'conversations' not in data:
                    errors.append(f"Line {line_num}: Missing 'conversations' field")
                elif len(data['conversations']) != 2:
                    errors.append(f"Line {line_num}: Should have exactly 2 conversations")
                else:
                    human, gpt = data['conversations']
                    if human.get('from') != 'human' or gpt.get('from') != 'gpt':
                        errors.append(f"Line {line_num}: Invalid conversation roles")

                if 'modality' not in data:
                    warnings.append(f"Line {line_num}: Missing 'modality' field")
                elif data['modality'] not in ['ct', 'mri', 'xray', 'ultrasound', 'fundus', 'pathology', 'endoscopy']:
                    warnings.append(f"Line {line_num}: Unknown modality: {data['modality']}")

                # Check if image exists
                if 'image' in data:
                    img_path = Path(data['image'])
                    if not img_path.exists():
                        errors.append(f"Line {line_num}: Image not found: {img_path}")
                    else:
                        try:
                            img = Image.open(img_path)
                            if img.size[0] < 224 or img.size[1] < 224:
                                warnings.append(f"Line {line_num}: Image too small: {img.size}")
                        except Exception as e:
                            errors.append(f"Line {line_num}: Cannot load image: {e}")

            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON: {e}")

    return errors, warnings

# Usage
errors, warnings = validate_medgemma_dataset('medical_dataset.jsonl')
print(f"Found {len(errors)} errors and {len(warnings)} warnings")
```

## 9. Next Steps

1. **Chuẩn bị dữ liệu** theo format trên
2. **Tạo file config** mới cho dataset của bạn:
   ```python
   # config/_base_/dataset/vietnamese_medical.py
   vietnamese_medical = dict(
       type='ConcatDataset',
       datasets=[
           dict(
               type='MedicalDataset',  # Need to create this class
               filename='data/vietnamese_medical.jsonl',
               image_folder='data/medical_images/',
               template_file='config/_base_/dataset/template/vietnamese_medical.json'
           )
       ]
   )
   ```

3. **Update main config** để sử dụng dataset mới:
   ```python
   # In medgemma_16gb_medical.py
   _base_ = ['../_base_/models/medgemma.py',
             '../_base_/dataset/vietnamese_medical.py',
             '../_base_/train/medgemma_lora.py']
   ```

4. **Run training** với data của bạn!
   ```bash
   python mllm/pipeline/finetune.py config/training_configs/medgemma_16gb_medical.py
   ```