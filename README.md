# 🫁 Pneumonia Triage AI: From Detection to Queue Simulation

## The Problem
When many pediatric chest X-rays are waiting to be read, reading order matters. An AI model may help prioritize suspected pneumonia, but a pneumonia scan given a low score could wait longer than under first-in, first-out (FIFO) ordering.

We built a pneumonia-scoring model and a waiting-line simulation to investigate both the benefits and the costs of AI prioritization. **This is a student research prototype, not a clinical tool or medical advice.** A pneumonia score does not measure clinical urgency.

## The Solution: Two-Notebook Workflow
1. **`01_model.ipynb`**: Fine-tunes an ImageNet-pretrained **DenseNet-121**, evaluates it, generates **Grad-CAM++** heatmaps, saves model weights and scores, and launches a Gradio web app.
2. **`02_sim.ipynb`**: Simulates a hospital queue using exported scores to compare FIFO, SCORE, SCORE + CAP and PERFECT. In our working files, this notebook is also called **`02_queue.ipynb`**. The existing queue results below use an earlier model's CSV and must be rerun for DenseNet-121.

The latest model uses square padding before resizing to 224 × 224 pixels to preserve image proportions, random flips and small rotations during training, and no random augmentation during validation or testing. Training uses class-weighted cross-entropy, AdamW and a cosine learning-rate schedule over six epochs. The checkpoint with the highest validation AUROC is saved and reloaded before evaluation.

Images are grouped by patient IDs inferred from filenames before splitting: **4,395 training images, 821 validation images and 624 test images**. The notebook reports 2,420 training and 426 validation patient-ID groups. Filename grouping reduces a leakage risk, but does not establish that all patient overlap or duplicate images have been excluded, particularly across the test set.

---

## 📊 Model Performance

> **Why is Accuracy 80.6% and not 99%?**
> Accuracy and pneumonia recall measure different things. At the 0.50 cutoff, DenseNet-121 identifies **389 of 390 pneumonia scans**, giving **99.7% recall**, but incorrectly flags **120 of 234 normal scans**. Its overall accuracy is therefore **80.6%**. High recall does not mean that every prediction is correct or that the model is ready for clinical use.

**Final Metrics (@0.5 Threshold):**
- **Accuracy:** 80.6% (503 of 624 scans classified correctly)
- **Pneumonia Recall:** 99.7% (1 missed pneumonia scan out of 390)
- **False Alarms:** 120 (Out of 234 normal scans)
- **Specificity:** 48.7% (114 of 234 normal scans classified correctly)
- **Precision (PPV):** 76.4% on this test set

These figures come from the saved outputs in the supplied DenseNet-121 notebook. The best validation AUROC is **0.9996**; this is a validation result, not the test AUROC. The last epoch's validation accuracy is **99.1%**.

### Threshold Sweep
The following results are from the **validation set only: 821 scans, including 635 pneumonia and 186 normal scans**. They must not be confused with the test results above. The notebook retains a final cutoff of **0.50**.

| Threshold | Accuracy | Recall | Missed (FN) | False Alarms (FP) |
|-----------|----------|--------|-------------|-------------------|
| 0.10 | 99.0% | 99.7% | 2 | 6 |
| 0.20 | 99.1% | 99.4% | 4 | 3 |
| 0.30 | 99.1% | 99.4% | 4 | 3 |
| 0.40 | 99.3% | 99.4% | 4 | 2 |
| **0.50** | **99.1%** | **99.2%** | **5** | **2** |
| 0.60 | 98.8% | 98.7% | 8 | 2 |
| 0.70 | 98.8% | 98.7% | 8 | 2 |
| 0.75 | 98.5% | 98.4% | 10 | 2 |
| 0.80 | 98.4% | 98.3% | 11 | 2 |
| 0.85 | 98.3% | 98.0% | 13 | 1 |
| 0.90 | 97.8% | 97.3% | 17 | 1 |
| 0.95 | 96.7% | 95.7% | 27 | 0 |

---

## 🚑 Queue Simulation Results (The Proof)
**These are earlier-model simulation results, not proof of clinical benefit and not results for the new DenseNet-121 model.**

The verified earlier experiment used a CSV with MD5 `9d9a7f07ba09221aa65c2b31407a55c3`, containing **2 missed cases and 119 false alarms**. It simulated 500 shifts with a 25% chance of pneumonia per scan. Each rule used the same scans, arrival times and reading times within each run.

| Queue Rule | Avg Wait (Pneumonia Scans) | Average Within-Shift 95th-Percentile Wait |
|------------|---------------------------|------------------------------------------|
| **Standard FIFO** | **101.1 minutes** | 148.6 minutes |
| **AI Triage (SCORE)** | **12.4 minutes** | 44.5 minutes |
| SCORE + CAP (60 minutes) | 81.3 minutes | 142.7 minutes |
| PERFECT (true labels known) | 9.0 minutes | 27.3 minutes |

**Conclusion:** In that earlier simulation, SCORE reduced average pneumonia waits but increased the worst observed missed-case wait from **223.6 to 370.7 minutes**. The 60-minute priority rule reduced this worst observed wait to **223.6 minutes**, while increasing average pneumonia and missed-case waits compared with SCORE. Only **80 of 500 shifts** contained missed cases. A cap raises priority; it does not guarantee a maximum wait.

The new notebook reports test-score MD5 **`f0ee6e9cafc6640e6c459d0724b71961`**, with **1 missed case and 120 false alarms**. **Rerun the queue using that new CSV before claiming waiting-time results for DenseNet-121.** Similar overall accuracy does not make two score files interchangeable.

---

## 🖥️ The Web App (Gradio)
The Gradio app allows users to upload an X-ray, adjust the classification threshold, and see:
1. A model classification and separate normal/pneumonia scores. These are not a diagnosis, measured clinical risk or calibrated clinical confidence, even where the current interface uses those labels.
2. A **Grad-CAM++** heatmap showing regions contributing to the pneumonia score, using `model.features.denseblock4`. It does not establish an exact disease boundary or prove that the model avoids shortcuts.

The threshold slider changes the classification label; it does not change the raw scores or the ordering of a queue sorted by those scores. The model was trained on public images of children aged 1–5 from one hospital; performance on other populations is unknown. Use public, de-identified sample images for demonstrations.

*(Temporary Gradio link from the supplied notebook: **[Open demo](https://c9abac0dc7b90518bb.gradio.live)**. This link requires the serving session to remain active and may expire. Permanent hosting has not been confirmed.)*

## 🚀 How to Run
1. Open the latest DenseNet-121 version of `01_model.ipynb` in Google Colab. Install any missing dependencies, including PyTorch, torchvision, kagglehub, scikit-learn, pandas, NumPy, matplotlib and Pillow. The notebook installs Gradio and grad-cam in its demo cells.
2. Run the notebook cells in order. The default output folder is **`./PneumoniaTriage`**, relative to the runtime's working directory. Download or persist **`densenet121_pneumonia.pth`**, **`val_scores.csv`** and **`test_scores.csv`** before the runtime resets. To reuse an existing trained model, restore its weights and run the loading/demo cells instead of retraining. Keep the matching `SquarePad` and `eval_tf` preprocessing.
3. Open the queue notebook (`02_sim.ipynb` in this repository, or `02_queue.ipynb` in the working files), set its input path to the **new** `test_scores.csv`, verify the counts and fingerprint, then rerun all simulations and charts. Update the website, report and video with results from that same run. Training uses seed 42, but exact repeatability across environments is not guaranteed; the training code also builds index lists by iterating over sets.

**Sources and acknowledgments:** [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia); [original Kermany et al. data, CC BY 4.0](https://doi.org/10.17632/rscbjbr9sj.2); Kermany et al. (2018), *Cell*; [Baltruschat et al., queue-prioritization study](https://doi.org/10.1007/s00330-020-07480-7); Huang et al., *Densely Connected Convolutional Networks*; Chattopadhay et al., *Grad-CAM++*; [pytorch-grad-cam implementation](https://github.com/jacobgil/pytorch-grad-cam). Built by **Vova and Danyal**, with coding, explanation and writing assistance from Claude and ChatGPT.
