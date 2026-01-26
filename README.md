# Rapid Detector

Minimalist object detection using SAM3. Create custom detectors with visual examples - no training required.

## Quick Start

```bash
pip install -e .
rapid-detector
```

Open http://localhost:8000

## Step-by-Step Labeling

1. **Upload images** - drag & drop your reference photos
2. **Name your detector** - type object class (e.g. "cookie", "car", "bottle")
3. **Select examples to detect** - green tool, draw boxes around target objects
4. **Select examples to exclude** - red tool, draw boxes around similar objects you want to exclude
5. **Submit annotations** - click submit to train the detector
6. **Test detection** - see results instantly on your images
7. **Copy API code** - get Python code to use your detector

## API Usage

```python
import requests

with open("test.jpg", "rb") as f:
    response = requests.post("http://localhost:8000/detect/cookie", files={"image": f})

results = response.json()
print(f"Found {len(results['boxes'])} cookies")
```

## Features

- **Zero-shot detection** - SAM3 foundation model + text prompts
- **Visual examples** - improve accuracy with positive/negative examples
- **Instant API** - REST endpoints for any detector

## Requirements

- Python 3.8+
- PyTorch
- GPU required

## License
This project uses Meta's Segment Anything Model (SAM 3). 
SAM 3 is licensed under the SAM License - see SAM3_LICENSE.txt
