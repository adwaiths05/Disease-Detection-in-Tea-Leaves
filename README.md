# 🍃 Tea Leaf Disease Classification with Grad-CAM 

This project trains a **ConvNeXt** model to classify diseases in tea leaves and uses **Grad-CAM** to visualize which regions of the leaf contribute most to the model’s predictions.

---

## 📂 Dataset

The dataset used is:

```
/kaggle/input/identifying-disease-in-tea-leafs/tea sickness dataset/
```

Organized into subfolders, one per disease class (e.g., `white spot`, `anthracnose`, etc.).

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install torch torchvision matplotlib opencv-python
```

---

## 🚀 Training the Model

1. Load ConvNeXt:

```python
import timm
model = timm.create_model("convnext_tiny", pretrained=True, num_classes=len(class_names))
```

2. Train or fine-tune using your dataset.

---

## 🔥 Grad-CAM Visualization

The Grad-CAM implementation hooks into the **last convolutional layer** of ConvNeXt to extract activations and gradients.

### Example function call:

```python
predicted_class = predict_with_gradcam(
    img_path="/kaggle/input/identifying-disease-in-tea-leafs/tea sickness dataset/white spot/UNADJUSTEDNONRAW_thumb_8f.jpg",
    model=model,
    class_names=class_names,
    device=DEVICE
)
print("✅ Predicted class:", predicted_class)
```

### Output:

* **Left**: Original tea leaf image
* **Right**: Grad-CAM heatmap overlay showing regions important for classification

---

## 📊 Grad-CAM Details

* **Target layer**: `features.7.2.block.0`

  * This is the **last convolutional layer** in ConvNeXt (768 channels).
* **Weights**: Computed as average pooled gradients.
* **CAM**: Weighted sum of activations, ReLU applied, resized to input image.

---

## 🛠️ Customization

* If you change the model (e.g., ConvNeXt-Small or ConvNeXt-Base), update the **target layer**:

  ```python
  target_layer = dict(model.named_modules())["features.7.2.block.0"]
  ```
* Or use an **auto-selection script** to always pick the last Conv2d layer.

---

## ✅ Results

Grad-CAM helps to:

* Verify if the model is focusing on diseased regions of tea leaves.
* Provide explainability to model predictions.

---

## 📌 References

* [ConvNeXt Paper (Facebook AI)](https://arxiv.org/abs/2201.03545)
* [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
* [timm library](https://github.com/huggingface/pytorch-image-models)
