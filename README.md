# Pneumonia-Triage
Pneumonia Triage
Track: Health
Goal: Which chest X-ray should the doctor read first?

The problem
Small hospitals often have one doctor and many chest X-rays waiting. When a doctor has more patients than time, they must decide where to look first. We built an AI that scans each X-ray for pneumonia so the most urgent scans get read first, and we tested how a doctor's waiting line is affected when the AI makes a mistake.

What we built
A model that takes a chest X-ray image and outputs a pneumonia score from 0 to 1. We simulated an 8-hour doctor shift to see if sorting by this score actually saves time compared to first-come, first-served, and what happens to the patients the AI misses.

What we added
We started with a standard pneumonia classification model, but we added a waiting-line simulation to ask a different question: How much do a model's misses hurt the queue, and does a maximum-wait cap fix it? We applied a published hospital simulation idea (Baltruschat et al., 2020) to pneumonia X-rays.

How it works
Data: We used the Kaggle Chest X-Ray Images (Pneumonia) dataset (Kermany et al., 2018). It contains 5,216 training images (split 85% train / 15% validation) and 624 test images from one children's hospital in Guangzhou (ages 1-5).
Model: We fine-tuned a pretrained ResNet-18 model using PyTorch and a T4 GPU on Google Colab.

Results
Model Performance (Held-out Test Set):

Baseline (Always guess pneumonia): 62.5% accuracy. It catches all sick kids but flags every healthy kid.

Our Model (ResNet-18): 82.5% accuracy, with a 99.0% pneumonia recall.

Where it fails: The model missed 4 out of 390 pneumonia cases (scores: 0.47, 0.32, 0.39, 0.11) and generated 105 false alarms. Because this is a triage tool, a missed case (false negative) is more dangerous than a false alarm (false positive).

Honest Note on Validation: During training, our validation accuracy reached 98.6%. However, because the Kaggle dataset contains multiple X-rays from the same child, a random split may leak patient data into the validation set. Therefore, we rely strictly on the fully unseen Test set, which gives an honest 82.5% score.

Queue Simulation Results:

What we'd do next
Test the model and queue simulation on X-rays from a different hospital to see if the scores hold up across different imaging machines.

Adjust the model's decision threshold to reduce the number of false alarms, reducing the burden on healthy patients.

Sources
Dataset: Kaggle Chest X-Ray Images (Pneumonia) by Paul Mooney. Original paper: Kermany et al., 2018 (Cell). License: CC BY 4.0.

Model: ResNet-18 pretrained on ImageNet, via torchvision.models.

Wait Cap Concept: Baltruschat et al., 2020 (European Radiology). DOI: 10.1007/s00330-020-07480-7.

Libraries: PyTorch, torchvision, pandas, numpy, matplotlib.

AI Tools: Claude (Anthropic) and ChatGPT (OpenAI) for coding assistance and queue simulation logic.

This project is a prototype and is not medical advice.
