# CURENET: Cardiac Health Diagnostic Tool

## Overview
This repository contains code and documentation for a novel clinician-in-the-loop, prompt-based Explainable Artificial Intelligence (XAI) tool developed for the interpretation of electrocardiogram (ECG) signals. The tool integrates multiple Deep Learning (DL) models with different XAI techniques to offer both accuracy and transparency in diagnosing cardiac abnormalities.

## Features
- **Deep Learning Models**: Trained on a comprehensive ECG dataset, our DL models achieve high precision in detecting cardiac abnormalities.
- **Explainable AI Techniques**: Integrated Gradients, Layer-wise Relevance Propagation (LRP), and DeepLift are employed to generate heatmaps, enhancing interpretability by highlighting decision-making regions within ECG signals.
- **Clinician-in-the-Loop Approach**: Our tool involves clinicians in the diagnostic process, providing prompt-based explanations for model decisions, and ensuring human expertise is utilized effectively.
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
3. You can use the sample data in the `test` dir to input ECG data for analysis.
4. View the generated heatmaps and diagnostic results.
5. Engage with the chatbot for additional inquiries.

## Streamlit Deployment

Our project is deployed on Streamlit and can be accessed via the following link:

[Streamlit Deployment](https://ecgxai-8ifypubpj9o4qgptjjq9lm.streamlit.app/)

## Paper Publication

Our project has been published in IEEE. The paper can be accessed on IEEE Xplore via the following link:

[IEEE Xplore](https://ieeexplore.ieee.org/document/10442356)

Alternatively, an openly available version of the paper can be found on ResearchGate:

[ResearchGate](https://www.researchgate.net/publication/378571216_CureNet_Improving_Explainability_of_AI_Diagnosis_Using_Custom_Large_Language_Models?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6ImhvbWUiLCJwYWdlIjoicHJvZmlsZSIsInByZXZpb3VzUGFnZSI6ImhvbWUiLCJwb3NpdGlvbiI6InBhZ2VDb250ZW50In19)

