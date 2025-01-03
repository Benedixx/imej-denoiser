# Denoising AutoEncoder (DAE) for Image Processing

This repository contains a Denoising AutoEncoder (DAE) model built for image processing tasks. The model is designed to remove noise from images, improving their quality and preparing them for further analysis. The implementation is efficient and leverages deep learning to achieve high performance.

## Features
- **Noise Removal**: Automatically denoises images with high accuracy.
- **Advanced Architecture**: Built with state-of-the-art components such as:
  - **Residual Blocks**: Inspired by ResNet for efficient feature extraction.
  - **Skip Connections**: Ensure better gradient flow during training.
  - **Attention Mechanisms**: Enhance model focus on important regions of the image such as edges and small text.
- **Customizable**: Easily adaptable for different datasets and noise types.
- **Optimized Loss Functions**: Combines multiple loss functions to improve performance:
  - **Perceptual Loss**: Ensures preservation of color and structural details using a pre-trained VGG16 network.
  - **Reconstruction Loss**: Uses a combination of Huber (Smooth L1) and L1 loss to minimize pixel-wise differences between the denoised and original images.
  - **SSIM Loss**: Enhances the structural similarity of the denoised images.

## Loss Function Details
The total loss function used during training combines various loss components as follows:

```python
from torchvision.models import vgg16

# Load pre-trained VGG16 model
vgg = vgg16(pretrained=True).features[:16].to(DEVICE).eval()

# Freeze VGG parameters
for param in vgg.parameters():
    param.requires_grad = False

def perceptual_loss(output, target):
    """Calculate perceptual loss using VGG16"""
    output_features = vgg(output)
    target_features = vgg(target)
    return F.mse_loss(output_features, target_features)

def loss_func(output, target):
    huber_loss = F.smooth_l1_loss(output, target, reduction='mean')
    l1_loss = F.l1_loss(output, target, reduction='mean')
    ssim_loss = 1 - ssim(output, target, data_range=1)
    perc_loss = perceptual_loss(output, target)
    total_loss = 0.6 * huber_loss + 0.3 * l1_loss + 0.05 * ssim_loss + 0.05 * perc_loss
    return total_loss
```

## Installation
Before setting up the environment, make sure the [UV package manager](https://docs.astral.sh/uv/) is installed on your machine or just install using pip
```bash
pip install uv
```
To set up the environment, there are two ways:

### Use Container (TensorRT supported)
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Benedixx/imej-denoiser.git
   cd imej-denoiser
   ```

2. **Build image**
   ```bash
   dockerbuild -t imej-denoiser .
   ```

3. **Run container**
   ```bash
   docker run -d --name imej-denoiser-container --gpus all -v %cd%:/imej-denoiser -p 8000:8000 imej-denoiser
   ```

### Install on venv (PyTorch inference)
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Benedixx/imej-denoiser.git
   cd imej-denoiser
   ```

2. **Install Dependencies**
   Use the UV package manager to create venv and install all required dependencies:
   ```bash
   uv sync
   ```
   This command will automatically resolve and install all necessary packages listed in `pyproject.toml`.

## Usage

1. **Prepare the Dataset** <br>
   Download the images from my kaggle dataset,
   Ensure the images are in the `data/` directory.
   ```bash
   curl -L -o ./data/imej-denoiser.zip \
   https://www.kaggle.com/api/v1/datasets/download/novebrititoramadhani/imej-denoiser

   unzip ./data/imej-denoiser.zip -d ./data/
   ```

2. **Train the Model** <br>
   Run the training script with the following arguments:
   ```bash
   python train.py --data_path <path_to_dataset> \
                   --batch_size <batch_size> \
                   --epochs <num_epochs> \
                   --n_epoch_stop <early_stopping_epochs> \
                   --model_path <output_model_path>
   ```
   **Arguments:**
   - `--data_path`: Path to the dataset. Default is `DATA_PATH`.
   - `--batch_size`: Batch size for training. Default is `BATCH_SIZE`.
   - `--epochs`: Number of training epochs. Default is `EPOCHS`.
   - `--n_epoch_stop`: Maximum epochs to wait for improvement during early stopping. Default is `N_EPOCHS_STOP`.
   - `--model_path`: Path to save the trained model. Default is `best_dae_model.pth`.

3. **Convert .pth model to .trt** (Optional) <br>
   You can convert the .pth model to .trt if you follow container installment for faster inference and memory efficient on your cuda device by running this script:

   1. Convert .pth to .onnx
   ```bash
   python torch2onnx.py
   ```
   2. Convert .onnx to .trt
   ```bash
   trtexec --onnx=dae_model.onnx --saveEngine=model.trt --explicitBatch --optShapes=input:1x3x256x256 --maxShapes=input:1x3x256x256
   ```

4. **Do inference** <br>
   Inference the trained model by modifying the input on the `inference.py` for .pth model or `inference_trt.py` for .trt model.

## Contributing
Feel free to fork this repository and submit pull requests. Contributions are welcome for:
- Improving the model architecture.
- Adding support for new noise types.
- Optimizing the training process.

## License
This project is licensed under the "Do What the Fuck You Want To" License. See the `LICENSE` file for details.
