# Rapid Detector

Minimalist zero-shot object detection using SAM3. Create custom detectors with visual examples - no training required.

⚠️ **Early alpha** - interfaces may change. [Feedback welcome!](https://github.com/tgeorgy/rapid-detector/issues)

## Great for

- Rapid prototyping for computer vision projects
- Custom object detection without ML expertise
- Production APIs that improve on-the-fly - fix failures by adding examples, no redeployment needed
- One-off detection tasks

## vs Roboflow/YOLO

- Built-in annotation tools
- Zero training time - results in seconds
- Live refinement - update deployed APIs by adding examples through the UI
- Local/private data - nothing leaves your machine
- Open source
- Tradeoff: Requires GPU

## Quick Start

You need to accept SAM3 license terms here https://huggingface.co/facebook/sam3
Setup access to huggingface hub here https://huggingface.co/docs/huggingface_hub/quick-start#quickstart

```bash
pip install -e .
python start_services.py
```

Open http://localhost:8000

## Step-by-Step Labeling

1. **Upload images** - drag & drop your reference photos
2. **Name your detector** - type object class (e.g. "cookie", "car", "bottle")
3. **Select examples to detect** - *Include* button to draw boxes around target objects
4. **Select examples to exclude** - *Exclude* button to draw boxes around similar objects you want to exclude
5. **Submit annotations** - click submit to update the detector
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
- **Visual examples** - improve accuracy with Include/Exclude examples
- **Instant API** - REST endpoints for any detector

## Requirements

- Python 3.8+
- PyTorch
- GPU required

## License
This project uses Meta's Segment Anything Model (SAM 3). 
SAM 3 is licensed under the SAM License - see SAM3_LICENSE.txt
