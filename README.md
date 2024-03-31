# Cardiac Health Diagnostic Tool

## Overview
This repository contains code and documentation for a novel clinician-in-the-loop, prompt-based Explainable Artificial Intelligence (XAI) tool developed for the interpretation of electrocardiogram (ECG) signals. The tool integrates multiple Deep Learning (DL) models with different XAI techniques to offer both accuracy and transparency in diagnosing cardiac abnormalities.

## Features
- **Deep Learning Models**: Trained on a comprehensive ECG dataset, our DL models achieve high precision in detecting cardiac abnormalities.
- **Explainable AI Techniques**: Integrated Gradients, Layer-wise Relevance Propagation (LRP), and DeepLift are employed to generate heatmaps, enhancing interpretability by highlighting decision-making regions within ECG signals.
- **Clinician-in-the-Loop Approach**: Our tool involves clinicians in the diagnostic process, providing prompt-based explanations for model decisions, ensuring human expertise is utilized effectively.
- **Chatbot Integration**: A pre-trained language model and an innovative image retrieval feature power a chatbot, offering an intelligent conversational agent for medical inquiries related to cardiac health.

## Installation
1. Clone the repository:
```bash
   git clone https://github.com/TejoVK/ECG_XAi.git
   ```
2. Install dependencies:
```bash
    pip install -r requirements.txt
    ```

## Usage
1. Navigate to the `ge_final` directory.

2. Run the main script:

```bash
   python finalcode.py
   ```
3. You can use the sample data in the test dir to input ECG data for analysis.
4. View the generated heatmaps and diagnostic results.
5. Engage with the chatbot for additional inquiries.
