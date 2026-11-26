# 👄 Real-Time Lip Contour Reconstruction and Model Comparison

## Introduction

The **Real-Time Lip Contour Reconstruction and Model Comparison** system is a computer vision application designed to detect, reconstruct, analyze, and measure lip contours from a live webcam feed.

The system uses **MediaPipe Face Mesh** to detect facial landmarks and extract the inner and outer lip contours. It then calculates lip measurements such as perimeter, width, height, area, length ratio, aspect ratio, and approximate measurements in centimeters.

The project also includes **PCA-based feature processing**, visibility and occlusion analysis, hand detection, real-time FPS monitoring, and comparison of different model variants through a web-based interface.

---

## Methods

### 1. Real-Time Video Capture

The system captures video directly from the computer's webcam using OpenCV.

The camera is configured for:

- 640 × 480 resolution
- 30 FPS target
- Continuous real-time frame processing

Each frame is processed and displayed through a Flask web application.

### 2. Image Enhancement

Before detecting facial landmarks, the input frame is enhanced using **CLAHE (Contrast Limited Adaptive Histogram Equalization)**.

The process includes:

- Conversion from BGR to LAB color space
- CLAHE enhancement of the luminance channel
- Reconstruction of the enhanced image
- Conversion to RGB for MediaPipe processing

This improves the quality of the input used for facial landmark detection.

### 3. Facial Landmark Detection

The project uses **MediaPipe Face Mesh** with refined landmarks to detect the face and identify lip regions.

The system:

- Detects one face at a time
- Uses refined facial landmarks
- Extracts predefined inner-lip landmark points
- Extracts the outer lip region from MediaPipe's lip connections
- Constructs ordered inner and outer contours

The extracted contours are used for further measurement and analysis. 

### 4. Lip Contour Reconstruction

Two major contours are reconstructed:

- **Inner Lip Contour**
- **Outer Lip Contour**

The outer contour is obtained using the convex hull of the detected lip landmark points, while the inner contour is constructed from selected inner-lip landmarks.

The points are ordered according to their angle around the contour center to create a consistent contour representation. 

### 5. Lip Measurement and Metrics

The system calculates multiple geometric measurements from the reconstructed contours.

The calculated metrics include:

- Outer lip perimeter
- Inner lip perimeter
- Outer lip width
- Outer lip height
- Inner lip width
- Inner lip height
- Outer lip area
- Inner lip area
- Inner-to-outer length ratio
- Inner-to-outer area ratio
- Outer aspect ratio
- Inner aspect ratio

Pixel measurements are also converted into approximate centimeter values using a configurable pixels-per-centimeter conversion factor.

### 6. Feature Extraction

The system creates feature vectors by pairing points from the inner and outer contours.

For each pair, it calculates:

- Normalized X distance
- Normalized Y distance
- Euclidean distance

A total of **20 contour pairs** are used for feature generation.

### 7. PCA Processing

Principal Component Analysis (**PCA**) is used to process the extracted contour features.

The system first collects a warm-up buffer of **80 frames**. After sufficient frames are collected, the feature data is normalized and PCA is initialized with **33 components**.

The interface also displays PCA initialization progress in real time. 

### 8. Visibility and Occlusion Detection

The project contains a visibility scoring system to determine whether the lip region is clearly visible.

The visibility score considers:

- Edge strength
- Texture variance
- Color consistency
- Contour smoothness
- Contour size consistency

The system also attempts to identify hand occlusion using skin-color and texture characteristics. When a hand is detected over the lip region, the visibility score is strongly reduced and the interface displays an occlusion warning.

### 9. Model Comparison

The Generalization mode compares four model variants:

- Base Model
- PS Model
- MS Model
- GR Model

Each model variant receives a live quality score derived from the current visibility measurement. The interface displays the scores, quality bars, and lip measurements for comparison.

### 10. Real-Time Visualization

The application provides three visualization modes:

#### 3×3 Grid View

Displays:

- Live face
- Occlusion status
- Processing pipeline
- Outer contour
- Inner contour
- Lip metrics
- Feature distances
- Combined visualization
- Detection information

#### Feature Analysis

Provides a detailed side-by-side visualization of:

- Inner contour
- Outer contour
- Feature distances
- Lip dimensions
- Lip areas
- Visibility score

#### Generalization

Provides:

- Current live frame
- Base model score
- PS model score
- MS model score
- GR model score
- Captured frame comparisons

These three modes can be switched directly from the web interface. 

---

## Output

The application provides a real-time browser-based dashboard containing:

- Live webcam video
- Reconstructed inner and outer lip contours
- Lip length measurements
- Lip width and height measurements
- Lip area measurements
- Length and area ratios
- Visibility percentage
- Hand occlusion detection
- PCA progress
- Real-time FPS
- Frame count
- Feature distance visualization
- Model comparison scores
- Captured-frame comparison

The dashboard runs through a Flask server at:

```text
http://localhost:5000
```

A webcam is required for real-time operation.

### Visualization Modes

| Mode | Description |
|---|---|
| **3×3 Grid View** | Complete real-time lip reconstruction and analysis |
| **Feature Analysis** | Detailed inner/outer contour and feature measurements |
| **Generalization** | Comparison of Base, PS, MS, and GR model variants |

### Keyboard Shortcuts

| Key | Function |
|---|---|
| `1` | Grid View |
| `2` | Feature Analysis |
| `3` | Generalization |
| `C` | Capture Frame in Generalization mode |

The web interface also provides buttons for changing modes and capturing or clearing frames.

---

## Conclusion

The **Real-Time Lip Contour Reconstruction and Model Comparison** project combines computer vision, facial landmark detection, geometric analysis, dimensionality reduction, and real-time web visualization into a single application.

The system is capable of reconstructing inner and outer lip contours from live webcam frames, calculating detailed lip measurements, monitoring visibility, detecting possible hand occlusion, tracking processing performance, and comparing different model variants.

The Flask-based interface makes the system interactive by providing multiple visualization modes, real-time statistics, keyboard shortcuts, and frame capture functionality.

Overall, the project demonstrates how **OpenCV, MediaPipe, PCA, NumPy, and Flask** can be integrated to build a real-time lip analysis and contour reconstruction system.

---

## Technologies Used

- **Python**
- **OpenCV**
- **MediaPipe Face Mesh**
- **NumPy**
- **Scikit-learn**
- **PCA (Principal Component Analysis)**
- **Flask**
- **HTML**
- **CSS**
- **JavaScript**
- **Webcam / Real-Time Video Processing**

---

