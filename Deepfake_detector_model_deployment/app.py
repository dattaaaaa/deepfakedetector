from flask import Flask, render_template, redirect, request, url_for, send_file, jsonify, json
from werkzeug.utils import secure_filename

# Interaction with the OS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Used for DL applications, computer vision related processes
import torch
import torchvision

# For image preprocessing
from torchvision import transforms

# Combines dataset & sampler to provide iterable over the dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import numpy as np
import cv2

# To recognise face from extracted frames
import face_recognition

# Autograd: PyTorch package for differentiation of all operations on Tensors
# Variable are wrappers around Tensors that allow easy automatic differentiation
from torch.autograd import Variable

import time
import sys

# 'nn' Help us in creating & training of neural network
from torch import nn

# Contains definition for models for addressing different tasks i.e. image classification, object detection e.t.c.
from torchvision import models

from skimage import img_as_ubyte
import warnings
warnings.filterwarnings("ignore")

UPLOAD_FOLDER = 'Uploaded_Files'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'mkv'}

app = Flask(__name__, template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Creating Model Architecture
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()

        # returns a model pretrained on ImageNet dataset
        model = models.resnext50_32x4d(pretrained=True)

        # Sequential allows us to compose modules nn together
        self.model = nn.Sequential(*list(model.children())[:-2])

        # RNN to an input sequence
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)

        # Activation function
        self.relu = nn.LeakyReLU()

        # Dropping out units (hidden & visible) from NN, to avoid overfitting
        self.dp = nn.Dropout(0.4)

        # A module that creates single layer feed forward network with n inputs and m outputs
        self.linear1 = nn.Linear(2048, num_classes)

        # Applies 2D average adaptive pooling over an input signal composed of several input planes
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape

        # new view of array with same data
        x = x.view(batch_size*seq_length, c, h, w)

        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


im_size = 112

# std is used in conjunction with mean to summarize continuous data
mean = [0.485, 0.456, 0.406]

# provides the measure of dispersion of image grey level intensities
std = [0.229, 0.224, 0.225]

# Often used as the last layer of a nn to produce the final output
sm = nn.Softmax(dim=1)  # Added dim=1 to fix softmax dimension issue

# Normalising our dataset using mean and std
inv_normalize = transforms.Normalize(mean=-1*np.divide(mean, std), std=np.divide([1,1,1], std))

# For image manipulation
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0,1)
    cv2.imwrite('./2.png', image*255)
    return image

# For prediction of output  
def predict(model, img, path='./'):
    # use this command for gpu    
    # fmap, logits = model(img.to('cuda'))
    fmap, logits = model(img.to('cpu'))  # Explicitly using CPU
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item()*100
    print('confidence of prediction: ', confidence)
    return [int(prediction.item()), confidence]

# To validate the dataset
class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    # To get number of videos
    def __len__(self):
        return len(self.video_names)

    # To get number of frames
    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            if frame is None:
                continue
                
            try:
                faces = face_recognition.face_locations(frame)
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                # If no face is detected, use the original frame
                pass
            
            if self.transform:
                try:
                    frame_t = self.transform(frame)
                    frames.append(frame_t)
                except:
                    # Skip problematic frames
                    continue
                    
            if len(frames) == self.count:
                break
                
        # Check if we have enough frames
        if len(frames) == 0:
            # Create a dummy frame if no frames were processed
            dummy_frame = np.zeros((im_size, im_size, 3), dtype=np.uint8)
            frames = [self.transform(dummy_frame) for _ in range(self.count)]
        elif len(frames) < self.count:
            # If we don't have enough frames, duplicate the last frame
            last_frame = frames[-1]
            while len(frames) < self.count:
                frames.append(last_frame)
                
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    # To extract number of frames
    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        if not vidObj.isOpened():
            print(f"Error: Could not open video file {path}")
            return
            
        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image
        vidObj.release()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detectFakeVideo(videoPath):
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    path_to_videos = [videoPath]

    try:
        video_dataset = validation_dataset(path_to_videos, sequence_length=20, transform=train_transforms)
        # use this command for gpu
        # model = Model(2).cuda()
        model = Model(2)
        
        # Check if model directory exists, if not create it
        if not os.path.exists('model'):
            os.makedirs('model')
            
        # Check if model file exists
        path_to_model = 'model/df_model.pt'
        if not os.path.exists(path_to_model):
            return [0, 0, "Model file not found. Please ensure 'model/df_model.pt' exists."]
            
        model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        model.eval()
        
        with torch.no_grad():
            prediction = predict(model, video_dataset[0], './')
            
        return prediction
    except Exception as e:
        print(f"Error in detection: {str(e)}")
        return [0, 0, str(e)]

@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Check if the request has a file part
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
        
    video = request.files['video']
    
    # If user submits empty form
    if video.filename == '':
        return jsonify({'error': 'No video selected'}), 400
        
    if not allowed_file(video.filename):
        return jsonify({'error': 'File type not allowed. Please upload a video file (mp4, avi, mov, wmv, mkv).'}), 400
        
    try:
        video_filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video.save(video_path)
        
        prediction = detectFakeVideo(video_path)
        
        if len(prediction) > 2:  # Error occurred
            result = {
                'status': 'error',
                'message': prediction[2]
            }
        else:
            if prediction[0] == 0:
                output = "FAKE"
            else:
                output = "REAL"
                
            confidence = round(prediction[1], 2)
            result = {
                'status': 'success',
                'output': output,
                'confidence': confidence
            }
            
        # Clean up - remove the uploaded file
        try:
            os.remove(video_path)
        except:
            pass
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 3000)), debug=True)
