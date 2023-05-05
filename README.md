# Realtime Object Detection using OpenCV

This repository contains code for simple real time object detection using OpenCV and Python. The code is tested using Webcam feed and realtime livestreams from a URL.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
- Python 3.7 or above
- OpenCV 4.7.0 or above (other versions might work, but not tested)

### Usage

1) Clone this repository

```bash
git clone https://github.com/pdragoi/opencv-object-detection.git
```

2) Install the requirements

```bash
pip install -r requirements.txt
```

3) Run the code

```bash
python main.py
```

The script will attempt to download the model weights and configurations. Existing files will be checked for the correct MD5 hash and will be skipped if the hash matches.

### Implemented models

- [x] YOLOv3
- [x] YOLOv2
- [x] EfficientDet-D0

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# TODO
- [ ] Add more models
- [ ] Add CLI arguments
- [ ] Add support for video files
