# Pneumonia-Triage
Pneumonia Triage Track: Health Goal: Which chest X-ray should the doctor read first?

The problem: Small hospitals often have one doctor and a long line of chest X-rays waiting to be read. When the doctor has patients than time they need to decide which scans to look at first. We built an AI model that scans each chest X-ray and gives it a pneumonia score from 0 to 1. The goal was to help the doctor focus on the urgent cases first. We also tested what happens when the AI makes mistakes—like missing a case of pneumonia—and how that affects the waiting line of patients.

What we built: A model that takes a chest X-ray image and outputs a pneumonia score between 0 and 1. We ran a simulation of an 8-hour doctor shift to compare two methods: sorting X-rays by the AI’s score versus a come first-served approach. We wanted to see if the AI improves efficiency and how bad it is when the AI misses a case.

What we added: We started with a pneumonia detection model.. We added a waiting-line simulation to answer a different question: How much do the model’s mistakes hurt the overall system?. Does setting a maximum waiting time for each patient fix that problem? We used a hospital queue simulation method from a published study (Baltruschat et al. 2020). Applied it to chest X-rays for pneumonia triage.

How it works:

Data: We used the Kaggle Chest X-Ray Images (Pneumonia) dataset (Kermany et al. 2018). This dataset includes 5,216 images for training (split into 85% train and 15% validation) and 624 test images from a children’s hospital in Guangzhou. All patients were aged 1 to 5 years.

Model: We fine-tuned a pretrained ResNet-18 model using PyTorch. We ran the training on a T4 GPU in Google Colab.

Results: Model Performance (on the held-out test set):

- Baseline ( guess pneumonia): 62.5% accuracy. This method catches all pneumonia cases. Also flags all healthy kids as sick.

- Our Model (ResNet-18): 82.5% accuracy with a 99.0% pneumonia recall. That means the model found all the real pneumonia cases.

Where it fails: The model missed 4 out of 390 pneumonia cases. Their scores were low: 0.47, 0.32 0.39 and 0.11. The model also had 105 alarms—meaning it said pneumonia when there was none. Because this is a triage tool missing a case (a false negative) is much worse than a false alarm (a false positive).

Honest Note on Validation: During training our model reached 98.6% accuracy on the validation set.. The Kaggle dataset has multiple images from the same child. If we split the data randomly some patients might appear in both training and validation sets. That’s a problem. So we only trust the test set results. The final accuracy is 82.5%—a honest number.

Queue Simulation Results:

We compared two methods:

- First-come, first-served

- Sorting by the AI’s pneumonia score

We found that the AI-based triage reduced the waiting time for patients.. When the AI missed a real case it caused a delay for that patient. In the case the missed case waited longer than it should have. Setting a maximum wait time (a wait cap) helped prevent delays from getting too bad. Patients who didn’t get read quickly were bumped up the queue after a set time.

What we’d do

1. Test the model and queue simulation on X-rays from a hospital. We want to see if the scores work across imaging equipment and patient populations.

2. Adjust the model’s decision threshold. If we raise the threshold fewer false alarms happen. That reduces the number of kids being flagged.. We need to make sure we don’t miss more pneumonia cases in the process.

Sources:

- Dataset: Kaggle Chest X-Ray Images (Pneumonia) by Paul Mooney. Original paper: Kermany et al. 2018 (Cell). License: CC BY 4.0.

- Model: ResNet-18 pretrained on ImageNet via torchvision.models.

- Wait Cap Concept: Baltruschat et al., 2020 (European Radiology). DOI: 10.1007/s00330-020-07480-7.

- Libraries: PyTorch, torchvision, pandas, numpy, matplotlib.

- AI Tools: Claude (Anthropic) and ChatGPT (OpenAI), for coding help and queue simulation logic.

This project is a prototype. It is not advice.
