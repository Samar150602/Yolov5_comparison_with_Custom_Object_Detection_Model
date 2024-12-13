# Object Detection with RESNET-50 and Custom Detection Head and Comparison with pre-trained Yolov5s, Yolov5l, Yolov5n and Yolov5m

In this project, we developed a custom object detection model using a ResNet-50 backbone and evaluated its performance against pre-trained YOLOv5 variants: YOLOv5n (nano), YOLOv5s (small), YOLOv5m (medium), and YOLOv5l (large). The evaluation was conducted on the COCO dataset using key performance metrics, including Recall, Precision, and Intersection over Union (IoU). These metrics provide a comprehensive understanding of the models' accuracy in detecting objects, the proportion of relevant objects detected, and the overlap between predicted and ground-truth bounding boxes. The comparison highlights the trade-offs between the custom model and YOLOv5's variants in terms of accuracy and computational efficiency, offering insights into their suitability for different real-world applications.

# Custom Model Architecture

The custom object detection model developed in this project uses a ResNet-50 backbone to extract robust image features and is tailored for object detection tasks on the COCO dataset. The model is designed to predict a fixed number of bounding boxes per image, with a classification head for object class probabilities and a regression head for bounding box coordinates. It utilizes CrossEntropy Loss for classification and Smooth L1 Loss for regression, with a mechanism to pad or truncate annotations to handle varying numbers of objects per image. While not as advanced or optimized as state-of-the-art models like YOLO, this model is built with modularity and simplicity in mind, serving as an effective learning tool and baseline for understanding the core concepts of object detection. The performance is compared against YOLOv5 variants to highlight strengths and areas for improvement in terms of precision, recall, and IoU.

#### 1. Backbone: ResNet-50
- **Purpose:**
  - Acts as the feature extractor to encode the input image into high-level spatial features.
- **Implementation:**
  - Pre-trained **ResNet-50** from `torchvision` is used, with the fully connected (FC) layers removed:
    ```python
    self.backbone = torchvision.models.resnet50(weights="DEFAULT")
    self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
    ```
  - Outputs a feature map of size `[batch_size, 2048, H/32, W/32]`, where `H` and `W` are the height and width of the input image.
- **Key Properties:**
  - The pre-trained backbone ensures robust feature extraction, leveraging knowledge from the ImageNet dataset.

#### 2. Feature Extraction
- The model uses a ResNet-50 backbone to extract high-level spatial features from input images.
- The fully connected layers of ResNet-50 are removed, and only the convolutional layers are retained.
- Features are pooled to a fixed spatial size of 7x7 using AdaptiveAvgPool2d.

#### 3. Classification Head
- The classification head is a series of fully connected layers designed to output class scores for each predicted bounding box:
    - Input: Flattened feature maps from the ResNet backbone, with dimensions [batch_size, 2048 * 7 * 7].
    - Layers:
        -A fully connected layer reduces dimensionality.
        -ReLU adds non-linearity, followed by dropout for regularization.
        -The final fully connected layer outputs logits for all classes for each bounding box, shaped as [batch_size, num_boxes_per_image, num_classes].
    - Output:
        - The logits are passed through a softmax function during inference to compute class probabilities for each bounding box.
    - Loss Function: CrossEntropy Loss compares the predicted class probabilities with ground truth class labels to train the model.

#### 4. Bounding Box Regression Head
- The regression head predicts the coordinates of bounding boxes for each detected object:
    - Input: Flattened feature maps from the ResNet backbone, similar to the classification head.
    - Layers:
        - A fully connected layer reduces dimensionality.
        - ReLU adds non-linearity, followed by dropout for regularization.
        - The final fully connected layer outputs 4 values ([x_min, y_min, x_max, y_max]) for each bounding box, shaped as [batch_size, num_boxes_per_image, 4].
    - Output:
        - The coordinates represent the top-left (x_min, y_min) and bottom-right (x_max, y_max) corners of the bounding box.
    - Loss Function: Smooth L1 Loss measures the difference between the predicted and ground truth bounding box coordinates, penalizing large errors less than Mean Squared Error.

### Loss Functions
In the custom object detection model, two separate loss functions are used to optimize the predictions during training:

Classification Loss:
    - Measures how well the model predicts the correct class for each bounding box.
Regression Loss: 
    - Measures how accurately the model predicts the bounding box coordinates for detected objects.

These two loss functions are combined into a single loss value that drives the model's learning process.

#### Intersection over Union (IoU)
Intersection over Union (IoU) is a common evaluation metric used in object detection tasks to measure the accuracy of predicted bounding boxes against ground truth bounding boxes. IoU quantifies how much the predicted bounding box overlaps with the ground truth box.

Definition
The IoU is calculated as the ratio of the area of overlap between the predicted and ground truth bounding boxes to the area of their union.

#### Precision and Recall
Precision and Recall are two fundamental metrics used to evaluate the performance of object detection models. They provide insights into the quality of the predictions by analyzing the balance between correct and incorrect predictions.
​
### Precision vs. Recall

| **Metric**   | **Focus**                   | **High Value Indicates**             |
|--------------|-----------------------------|--------------------------------------|
| **Precision** | Quality of predictions     | Most predictions are correct         |
| **Recall**    | Completeness of detections | Most ground truth objects are detected |


# YOLO Model
YOLO (You Only Look Once) is a family of state-of-the-art object detection models that are widely used for their speed and accuracy. YOLO models unify object detection into a single neural network that predicts bounding boxes and class probabilities in one forward pass. This makes YOLO models highly efficient, enabling real-time object detection.

### YOLOv5 Pretrained Models

YOLOv5 offers multiple pretrained models of different sizes, optimized for various use cases:

| **Model Variant** | **Description**                                    | **Performance**                   |
|--------------------|----------------------------------------------------|------------------------------------|
| **YOLOv5n**        | Nano version, optimized for low-resource devices. | Smallest and fastest.             |
| **YOLOv5s**        | Small version, balances speed and accuracy.       | Ideal for lightweight tasks.      |
| **YOLOv5m**        | Medium version, more accurate but slower.         | Suitable for balanced use cases.  |
| **YOLOv5l**        | Large version, high accuracy but resource-intensive. | For high-performance applications. |


