## Project Overview
This project classifies images of soybean crops into four categories: **soybean**, **grass**, and **broadleaf** and **soil** to detect the presence of different types of weeds (broadleaf vs grass weeds).

## Performance Summary
| Metric | Result |
| :--- | :--- |
| **Test Accuracy** | **99.74%** |
| **Test Loss** | **0.0105** |

*   **Framework:** TensorFlow / Keras
*   **Deployment:** Streamlit

    The model performed extremely well on both the validation and test set. It had a nearly perfect test accuracy, which indicated that the model achieved high generalization and is able to accurately differentiate soybean crops from common weeds. 

## Technical Implementation

### Data Preprocessing
- **Source:** [Weed Detection in Soybean Crops](https://www.kaggle.com) (~15,336 images).
- **Image Conversion:** Images were processed from TIFF to JPEG for TensorFlow compatibility.
- **Resizing:** High-resolution ($2672 \times 4000$) images were downscaled to **$224 \times 224$** to match the input requirements of the MobileNetV3 architecture.
- **Class Balancing:** Applied manual class weights during training to compensate for disproportionate image counts across the four categories.
- **Dataset Splitting:** Utilized an 85/15 split for training and testing, with a further 20% of the training data partitioned for real-time validation monitoring.

### Model Architecture

Instead of a standard CNN, the system leverages a MobileNetV3-Large backbone, optimized for mobile and web-based deployment:

- **Feature Extractor:** A pre-trained MobileNetV3-Large with frozen base layers to retain low-level edge and texture detection.
- **Global Average Pooling (GAP):** Used to reduce the spatial dimensions of the feature maps into a 1D vector, significantly reducing the number of trainable parameters.
- **Regularization Dropout (0.3):** Applied to the dense layers to prevent the model from overfitting on the specific field conditions of the training set.
- **Output Layer:** A Dense layer with 4-way Softmax activation, mapping features to the specific classes: Broadleaf, Grass, Soil, and Soybean.

### Training Strategy
- **Framework:** Built using TensorFlow 2.x and Keras for a modular, scalable pipeline.
- **Inference Stability:** Called the model with training=False during the prediction phase to ensure BatchNormalization layers used moving averages rather than mini-batch statistics.
- **Loss & Optimization:** Categorical Cross-Entropy was paired with the Adam optimizer for efficient gradient descent.
- **Convergence Control:** Implemented EarlyStopping monitoring val_loss with a patience of 3 epochs, ensuring the final model captured the "Best Weights" before any divergence occurred.

## Limitations

1. The model was only trained on UAV (drone) images, which look different than images taken with other cameras. The model may perform differently if non-UAV images are uploaded on the web-page. 

2. The model was only trained to identify broadleaf vs grass weeds, so it is unable to identify any other weeds such as woody or flowering weeds.
3. This model was only designed to detect weeds amongst well established soybean crops. In the original dataset, any images containing mostly soil were labeled soil, even if there were some small weeds present.
4. The model may not recognize damaged or diseased soybean plants, as these were not present in the training dataset.

## Project Impact

This project has to potential to improve efficiency of herbicide use in soybean crops. If modified for commercial use, this model could lower operational costs and minimize environmental footprint from excessive chemical use.

## How to Deploy
1. **Clone the repo** and ensure `soybean_weed_model.keras` is in the directory.
2. **Ensure that you have Python 3.10 or 3.12 installed.**
3. **Run the following commands to launch the streamlit application:**

chmod +x setup.sh

./setup.sh

