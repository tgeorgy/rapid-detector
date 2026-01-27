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

## Python Library Usage

Use the detector directly in Python without the API server:

```python
from rapid_detector import RapidDetector
from PIL import Image

# Initialize detector (loads SAM3 model)
detector = RapidDetector()

# Load an existing detector
detector.load_config('cookie')

# Run detection on an image
image = Image.open('test.jpg')
results = detector.run_inference('cookie', image, confidence_threshold=0.5)

# Results contain masks, boxes, and scores
print(f"Found {len(results['boxes'])} detections")
for i, (box, score) in enumerate(zip(results['boxes'], results['scores'])):
    print(f"Detection {i+1}: {box} (confidence: {score:.2f})")
```

### Creating a new detector programmatically

```python
from rapid_detector import RapidDetector
from PIL import Image

detector = RapidDetector()

# Create a new detector
detector.new_config('bottle')

# Add visual examples (bounding boxes on reference images)
reference_image = Image.open('reference.jpg')
boxes = [[100, 150, 200, 300], [400, 100, 500, 250]]  # [x1, y1, x2, y2]
labels = [True, True]  # True = include, False = exclude

detector.update_prompts('bottle', reference_image, boxes, labels)

# Save detector for later use
detector.save_config('bottle')

# Run inference
test_image = Image.open('test.jpg')
results = detector.run_inference('bottle', test_image)
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
