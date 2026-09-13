# Image Captioning with Encoder-Decoder Architectures on MS-COCO

This repository contains a comprehensive deep-learning implementation of an **Image Captioning System** developed for the Statistical Natural Language Processing course[cite: 5]. The system generates natural language descriptions for visual inputs by combining Convolutional Neural Networks (CNNs) for feature extraction and Recurrent Neural Networks (RNNs) for sequence generation, enhanced progressively with Teacher Forcing and Global Attention mechanisms.

---

## System Architecture & Methodology

### 1. Dataset & Data Splitting Strategy
* **Dataset:** A curated subset of the **MS-COCO 2014** training dataset comprising **30,000 unique images** and their corresponding descriptive captions.
* **Splitting:** 
  * **Training Split (80%):** 24,000 images.
  * **Validation Split (20%):** 6,000 images.
  * **Test Split:** An independent, completely unseen test set of **500 images** sampled randomly from validation for rigorous quantitative evaluation.

### 2. Encoder-Decoder Foundation
* **CNN Encoder (VGG16):** Pretrained VGG16 (excluding fully connected layers) extracts spatial feature maps of dimension $7 \times 7 \times 512$ from the last convolutional layer.
* **RNN Decoder (`UniversalDecoder`):** A flexible custom decoder supporting both LSTM and GRU cells with parameterized layer configurations (optimized using a **2-layer GRU** for performance-speed balance).

### 3. Experimental Phases
* **Phase 1 (Baseline):** Standard Encoder-Decoder architecture without attention or teacher forcing.
* **Phase 2 (Teacher Forcing):** Stabilizes training convergence by feeding the ground-truth target word from the dataset as the next step's input instead of the model's own prior prediction.
* **Phase 3 (Global Attention Mechanisms):** Integrates spatial attention scoring functions allowing the model to dynamically focus on relevant image regions:
  1. **Additive (Bahdanau) Attention**
  2. **Dot-Product (Luong) Attention**
  3. **Scaled Dot-Product Attention**

### Advanced Features & Bonus Implementations
* **Region Attention Networks (Bonus #2):** Incorporates a `region_projector` layer to capture finer spatial figure details within the attention mechanism.
* **Visual Interpretability (Bonus #3):** Logs attention weights per generated word to map and visualize where the model focuses on a $7 \times 7$ grid matrix during inference.

---

## Quantitative Results & Performance Analysis

All models were evaluated on the 500-item unseen test set using corpus-level **BLEU-1, BLEU-4, METEOR, and ROUGE-L** metrics.

| Architecture Strategy | BLEU-1 | BLEU-4 | METEOR | ROUGE_L |
| :--- | :---: | :---: | :---: | :---: |
| **Phase 1: Baseline (No TF, No Attn)** | 0.4313 | 0.0308 | 0.2205 | 0.3734 |
| **Phase 2: Teacher Forcing (2L GRU)** | **0.5703** | **0.1254** | **0.3479** | **0.4524** |
| **Phase 3: Bahdanau (Additive) Attention** | 0.5595 | 0.1215 | 0.3415 | 0.4451 |
| **Phase 3: Luong (Dot-Product) Attention** | 0.4522 | 0.0312 | 0.2317 | 0.3686 |
| **Phase 3: Scaled Dot-Product Attention** | 0.4023 | 0.0120 | 0.1881 | 0.3429 |

* **Key Takeaway:** Teacher Forcing accelerated grammar learning and boosted the BLEU-4 score by approximately **4x** compared to the baseline. Furthermore, Bahdanau's additive attention proved significantly superior in aligning image feature spaces with text tokens than dot-product variants.

---

## Model Interpretability & Attention Heatmaps

The visualization pipeline exposes the internal decision-making mechanics of the network[cite: 5]. For instance, when generating the caption *"a man in a suit is playing baseball"*, attention heatmaps confirm the model correctly shifts focus from the person (`man`) to attire (`suit`) and the overarching field dynamics (`playing baseball`).

---

## Tech Stack
* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Metrics & Evaluation:** NLTK (`corpus_bleu`, `meteor_score`), `rouge-score`
* **Environment:** Google Colab GPU runtime with Google Drive checkpoint integration

## How to Run
1. Open the provided Jupyter Notebooks in Google Colab to utilize GPU acceleration.
2. Mount your Google Drive to handle dataset downloads (MS-COCO 2014) and feature extraction backups (`.npy` files).
3. Execute the modular scripts sequentially: Feature Extraction $\rightarrow$ Phase 1 Baseline $\rightarrow$ Phase 2 Teacher Forcing $\rightarrow$ Phase 3 Attention Experiments & Evaluation.
