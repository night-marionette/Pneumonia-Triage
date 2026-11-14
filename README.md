# 🫁 Pneumonia Triage AI: From Detection to Queue Simulation

## The Problem
In pediatric radiology, missing a pneumonia case (False Negative) can be deadly. However, standard radiology queues are "First-In, First-Out" (FIFO), meaning a sick child waits behind perfectly healthy children. 

We built an end-to-end system that uses a Deep Learning model to triage X-rays, ensuring sick children are prioritized and seen up to **8x faster**.

## The Solution: Two-Notebook Workflow
1. **`01_model.ipynb`**: Trains a ResNet-18 model, evaluates it, generates Grad-CAM heatmaps, and launches a Gradio web app.
2. **`Model2.ipynb`**: Simulates a hospital queue using the model's predictions to prove the clinical impact of AI triage.

---

## 📊 Model Performance

> **Why is Accuracy 80.6% and not 99%?**
> Because in pediatric triage, missing a sick kid is deadly. We optimized the model for **99.5% Recall** (catching almost every sick kid). The trade-off is 119 false alarms. We proved via simulation that using this model means sick kids wait 12 minutes instead of 98 minutes.

**Final Metrics (@0.5 Threshold):**
- **Accuracy:** 80.6% 
- **Pneumonia Recall:** 99.5% (Only missed 2 out of 390 sick kids)
- **False Alarms:** 119 (Out of 234 healthy kids)

### Threshold Sweep
| Threshold | Accuracy | Recall | Missed (FN) | False Alarms (FP) |
|-----------|----------|--------|-------------|-------------------|
| **0.50** | **80.6%** | **99.5%** | **2** | **119** |
| 0.60 | 81.6% | 99.5% | 2 | 113 |
| 0.70 | 82.5% | 99.2% | 3 | 106 |
| 0.80 | 84.1% | 99.0% | 4 | 95 |
| 0.90 | 86.7% | 99.0% | 4 | 79 |

---

## 🚑 Queue Simulation Results (The Proof)
Using our `test_scores.csv` (hash verified: `9d9a7f07...`), we simulated 500 hospital shifts. 

| Queue Rule | Avg Wait (Sick Kids) | 95th Percentile Wait |
|------------|----------------------|----------------------|
| **Standard FIFO** | **98.6 minutes** | 148.2 mins |
| **AI Triage (SCORE)** | **12.3 minutes** | 44.5 mins |
| Perfect God-Mode (PERFECT) | 9.1 minutes | 27.3 mins |

**Conclusion:** Using the AI model to sort the queue gets sick children seen in ~12 minutes on average, compared to ~98 minutes with a standard first-in, first-out queue. The AI triage is almost as good as a perfect, all-knowing doctor.

---

## 🖥️ The Web App (Gradio)
The Gradio app allows users to upload an X-ray, adjust the triage threshold, and see:
1. The AI diagnosis (with exact probabilities).
2. A Grad-CAM heatmap showing exactly which part of the lung the AI is looking at.

*(Link to Hugging Face Space: **[INSERT YOUR HF SPACE LINK HERE]**)*

## 🚀 How to Run
1. Open `01_model.ipynb` in Google Colab.
2. Run all cells to train the model and launch the Gradio app.
3. Open `Model2.ipynb` and ensure the path to `test_scores.csv` is correct, then run all cells to generate the simulation charts.
