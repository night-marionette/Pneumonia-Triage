# 🫁 Pneumonia Triage AI: From Detection to Queue Simulation

## The Problem
When many pediatric chest X-rays are waiting to be read, reading order matters. An AI model may help prioritize suspected pneumonia, but a pneumonia scan given a low score could wait longer than under first-in, first-out (FIFO) ordering.

We built a pneumonia-scoring model and a waiting-line simulation to investigate both the benefits and the costs of AI prioritization. **This is a student research prototype, not a clinical tool or medical advice.** A pneumonia score does not measure clinical urgency.

## The Solution: Two-Notebook Workflow
1. **[`01_model.ipynb`](https://github.com/night-marionette/Pneumonia-Triage/blob/main/01_model.ipynb)**: Fine-tunes an ImageNet-pretrained **DenseNet-121**, evaluates it, generates **Grad-CAM++** heatmaps, saves model weights and scores, and launches a Gradio web app.
2. **[`02_sim.ipynb`](https://github.com/night-marionette/Pneumonia-Triage/blob/main/02_sim.ipynb)**: Simulates a hospital queue using the matching DenseNet-121 scores to compare FIFO, SCORE, SCORE + CAP and PERFECT.

**Training hardware:** We trained DenseNet-121 in **Google Colab using an NVIDIA T4 GPU**.

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

These figures match both the supplied DenseNet-121 notebook and the uploaded test-score CSV. The best validation AUROC is **0.9996**; this is a validation result, not the test AUROC. The last epoch's validation accuracy is **99.1%**.

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
**These results use the latest DenseNet-121 scores. They demonstrate behavior in a simulation, not clinical benefit.**

The uploaded CSV and the simulation's saved fingerprint match: MD5 **`f0ee6e9cafc6640e6c459d0724b71961`**, with **624 scans, 1 missed pneumonia case and 120 false alarms** at a 0.50 threshold.

We simulated **500 shifts**, using seeds **0–499**. At the base setting, each simulated scan has a **25% chance of being pneumonia**; this is an assumption, not a measured hospital rate. Scores are sampled with replacement from the appropriate class in the CSV. Each rule uses the same scans, arrival times and reading times within each run.

One doctor starts with **30 waiting scans**. Arrival gaps and reading times are exponential, averaging **5 minutes** and **4.5 minutes**, respectively. Arrivals stop at **480 minutes**, but reading continues until the queue is empty. Waiting time is arrival to the start of reading. These timings are experimental assumptions.

FIFO selects the oldest scan; SCORE selects the highest score. SCORE + CAP selects the oldest scan that has waited **more than 60 minutes**, otherwise the highest score. PERFECT selects true pneumonia first, then the oldest within each class. It uses labels unavailable in practice and is not an optimum for every metric. The 0.50 cutoff defines missed cases; it does not affect raw-score ordering.

| Queue Rule | Avg Wait (Pneumonia Scans) | Average Within-Shift 95th-Percentile Wait |
|------------|---------------------------|------------------------------------------|
| **Standard FIFO** | **101.1 minutes** | 148.6 minutes |
| **AI Triage (SCORE)** | **12.4 minutes** | 45.7 minutes |
| SCORE + CAP (60 minutes) | 81.6 minutes | 142.7 minutes |
| PERFECT (true labels known) | 9.0 minutes | 27.3 minutes |

**Spread across shifts:** The 5th–95th-percentile ranges for each shift's average pneumonia wait are **43.4–160.7 minutes** for FIFO, **5.3–22.2** for SCORE, **23.0–141.3** for SCORE + CAP and **4.5–16.2** for PERFECT. These describe variation across simulated shifts, not confidence intervals.

**Missed cases and the cost to normal scans:** Values below are minutes, shown as **mean [5th–95th percentile across contributing shifts]**. The last column is a single observed maximum, not an average.

| Queue Rule | Missed: Average Wait | Missed: Average Within-Shift Maximum | Normal: Average Wait | Missed: Worst Across All Runs |
|------------|----------------------|-------------------------------------|----------------------|-------------------------------|
| **FIFO** | 109.1 [32.4–193.8] | 109.9 [32.4–193.8] | 101.1 [44.4–158.2] | 223.6 |
| **SCORE** | 101.5 [5.3–298.5] | 102.6 [5.3–298.5] | 130.2 [55.8–207.9] | 321.2 |
| **SCORE + CAP (60)** | 119.6 [61.3–193.8] | 120.5 [61.3–193.8] | 107.5 [50.9–164.6] | 223.6 |
| **PERFECT** | 7.3 [0.2–22.3] | 7.5 [0.2–22.3] | 131.9 [55.8–209.9] | — |

The PERFECT single worst wait is omitted here because it is not printed in the notebook's numerical tables.

**Wait-cap comparison (25% pneumonia share):** Average waits are means across contributing shifts; all values are minutes.

| Priority Cap | Pneumonia: Average Wait | Missed: Average Wait | Missed: Average Within-Shift Maximum | Normal: Average Wait | Missed: Worst Across All Runs |
|--------------|-------------------------|----------------------|-------------------------------------|----------------------|-------------------------------|
| 30 minutes | 90.5 | 113.1 | 114.0 | 104.5 | 223.6 |
| 60 minutes | 81.6 | 119.6 | 120.5 | 107.5 | 223.6 |
| 120 minutes | 57.5 | 136.5 | 137.8 | 115.6 | 229.1 |

The notebook also tests **10%, 25% and 60% pneumonia shares**, with **10, 37 and 84 of 500 shifts**, respectively, containing missed cases. Missed-case summaries exclude shifts without a missed case. At the base 25% share, only **37 shifts** contribute, and every sampled miss comes from the **same one original test image**. Repeated sampling does not represent diverse model failures.

**Conclusion:** SCORE reduced average pneumonia waiting time from **101.1 to 12.4 minutes**, while normal scans waited longer (**101.1 to 130.2 minutes**). Although average missed-case wait fell from **109.1 to 101.5 minutes**, the worst observed missed-case wait rose from **223.6 to 321.2 minutes**. The 60-minute priority cap reduced that worst observed wait to **223.6 minutes**, but raised average missed-case wait to **119.6 minutes** and average pneumonia wait to **81.6 minutes**. **The cap trades faster prioritization for reduced extreme delays; it does not guarantee a maximum wait or improve every metric.**

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
3. Open **`02_sim.ipynb`** and run its code cells in order. It first checks `./PneumoniaTriage/test_scores.csv`, then `./test_scores.csv`; if neither exists, it downloads `test_scores.csv` from the repository. Google Drive is not required. Confirm **1 missed case, 120 false alarms and MD5 `f0ee6e9cafc6640e6c459d0724b71961`** before using the results above: an existing local file is reused, and the GitHub file can change. The final cell exports charts, numerical results and **`Pneumonia_Triage_Results.zip`** into `./PneumoniaTriage`. Keep the website, report and video consistent with this same run. Training uses seed 42, but exact repeatability across environments is not guaranteed; the training code also builds index lists by iterating over sets.

**Sources and acknowledgments:** [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia); [original Kermany et al. data, CC BY 4.0](https://doi.org/10.17632/rscbjbr9sj.2); Kermany et al. (2018), *Cell*; [Baltruschat et al., queue-prioritization study](https://doi.org/10.1007/s00330-020-07480-7); Huang et al., *Densely Connected Convolutional Networks*; Chattopadhay et al., *Grad-CAM++*; [pytorch-grad-cam implementation](https://github.com/jacobgil/pytorch-grad-cam). Built by **Vova and Danyal**, with coding, explanation and writing assistance from Claude, Google AI, Deepseek and ChatGPT.
