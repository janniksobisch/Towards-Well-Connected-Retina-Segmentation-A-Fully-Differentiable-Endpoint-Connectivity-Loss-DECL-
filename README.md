# Towards Well-Connected Retina Segmentation: A Fully Differentiable Endpoint Connectivity Loss (DECL)

This is the official repository for the paper **"Towards Well-Connected Retina Segmentation: A Fully Differentiable Endpoint Connectivity Loss (DECL)"**. 

This repository hosts the implementation of the DECL loss function, designed specifically for seamless integration within the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework.

## Installation & Usage

To use the **DECL (Endpoint Average Loss)** in your own nnU-Net v2 projects, you just need to drop two files into your local nnU-Net installation:

**1. Add the Loss Function** Copy the `end_distance_loss.py` file into your nnU-Net loss directory:
```bash
cp loss/end_distance_loss.py /path/to/your/nnUNet/nnunetv2/training/loss/ 
```

**2. Add the Custom Trainer** Copy the custom trainer file into your nnU-Net trainers directory:
```bash
cp nnUNetTrainerEndpointAverage.py /path/to/your/nnUNet/nnunetv2/training/nnUNetTrainer/
```

**3. Train the Model** Start your training using the newly added custom trainer by specifying it with the `-tr` flag:
```bash
nnUNetv2_train [DATASET_ID] [2d/3d_fullres] [FOLD] -tr nnUNetTrainerEndpointAverage
```
