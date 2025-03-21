# DeepFake Detector

A deep learning-based system for detecting manipulated (deepfake) videos. This project provides tools for training models to distinguish between real and fake videos, and includes deployable applications for real-time detection.

# Check it out

Try the latest deployment of the Deepfake detector here:
[Deepfake Detection App](https://deepfakedetection.up.railway.app)

## Project Overview

This repository contains a complete pipeline for deepfake detection:

1. **Training**: Jupyter notebooks for model training and evaluation
2. **Model Deployment**: Flask-based web application for deepfake detection
3. **Final Deployment**: Production-ready web application for user interaction

The system utilizes a combination of:
- ResNext50 for feature extraction
- LSTM for temporal feature analysis
- Face recognition for focusing on facial regions

## Directory Structure

- **Training/**: Contains Jupyter notebook for model training and testing
  - `Model_train.ipynb`: Main notebook for training the model
  - `Preprocess.py`: python file for preprocessing data

- **Deepfake_detector_model_deployment/**: Contains the main model deployment app
  - `app.py`: Flask application for video analysis
  - `templates/`: Contains the web interface
  - `Dockerfile`: For containerizing the application
  - `requirements.txt`: Dependencies for the application

- **Final_Deployment/**: Contains the production-ready application
  - `app.py`: Simplified Flask application
  - `templates/`: Contains the final user interface

## How It Works

The system follows these steps for deepfake detection:

1. Video upload through web interface
2. Frame extraction from the video
3. Face detection and extraction for each frame
4. Feature extraction using ResNext50
5. Temporal feature analysis using LSTM
6. Classification as real or fake with confidence score

## Model Architecture

The neural network architecture consists of:
- Feature Extractor: ResNext50_32x4d (pretrained on ImageNet)
- Sequence Modeling: LSTM
- Classification: Fully connected layer

## Dataset

The model was trained on a combination of dataset:
- FaceForensics++ (FF)

## Requirements

See `requirements.txt` for detailed dependencies. Major requirements:
- Python 3.12.x
- PyTorch
- OpenCV
- Flask
- Face Recognition
- dlib

## Installation and Setup

### Option 1: Using Docker

1. Clone the repository:
   ```
   git clone https://github.com/dattaaaaa/deepfakedetector.git
   cd DeepFakeDetector
   ```

2. Build and run the Docker container:
   ```
   cd Deepfake_detector_model_deployment
   docker build -t deepfake-detector .
   docker run -p 3000:3000 deepfake-detector
   ```

3. Access the application at `http://localhost:3000`

### Option 2: Manual Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/DeepFakeDetector.git
   cd DeepFakeDetector
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   cd Deepfake_detector_model_deployment
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python app.py
   ```

5. Access the application at `http://localhost:3000`

### Option 3: Use existing Docker-image
1. Pull the image and run it:
   ```
   docker pull bharshavardhanreddy924/deepfake_detection
   docker run -p 8080:8080 --name deepfake_detector bharshavardhanreddy924/deepfake_detection
   ```

## Training Your Own Model

1. Prepare your dataset with real and fake videos
2. Open `Training/Model_train.ipynb` in Jupyter Notebook or Google Colab
3. Follow the instructions in the notebook to train the model
4. The trained model will be saved as `df_model.pt`
5. Place the model in the `model/` directory of the deployment application

## Using the Web Interface

1. Access the application through your browser
2. Upload a video file (supported formats: mp4, avi, mov, wmv, mkv)
3. Click "Detect Deepfake"
4. Wait for the analysis to complete
5. View the result showing real/fake prediction with confidence score

## Model Performance

The model achieves:
- High accuracy in detecting manipulated videos
- Real-time performance for quick analysis
- Special focus on facial regions for better reliability

## Limitations

- Performance may vary depending on video quality
- Face detection might fail in low light or obscured faces
- Large videos may take longer to process





## Acknowledgments

- The developers of PyTorch, OpenCV, and Face Recognition
- Contributors to the deepfake detection research community
- Datasets used for training: FaceForensics++
