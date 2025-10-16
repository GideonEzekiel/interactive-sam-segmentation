#!/bin/bash

# Download the SAM ViT-B model checkpoint if it doesn't exist
if [ ! -f "sam_vit_b.pth" ]; then
    echo "Downloading SAM ViT-B checkpoint..."
    # Using curl as an alternative to wget for better compatibility
    curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -o sam_vit_b.pth
    echo "Download complete."
fi

# The deployment environment will run this script before starting the Python app.