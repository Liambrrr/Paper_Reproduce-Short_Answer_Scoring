# Paper_Reproduce-Short_Answer_Scoring

## Overview
GPT_inference.ipynb was adapted from [lan-j/SAS_GPT4](https://github.com/lan-j/SAS_GPT4).
Instead using GPT-4, this project was running with model Llama 3.1 405b instruct.

## Quick Start

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Pipeline

```bash
python step1_sample_data.py
# run all cells in GPT_inference.ipynb
# but CHANGE variable 'question' with respect to Q1 to Q10 in cell 3
python step6_evaluation.py
python step7_subject_report.py
```
## Step Instruction
Step 1 - Sample the data
For each question, randomly sample 100 responses, then add more until every label frequency  ≥ 10. This will be 1538 responses in total, across Q1-Q10. In paper, the numbers are: Q4=163, Q5=356, Q6=319, and 100 responses for each remaining question. They didn’t specifically mention about random sampling part. So, we can follow those question numbers, by double-checking label frequency.


Step 6 - Evaluation
For each question, calculate accuracy and QWK separately. Then, calculate average of the ten questions and report it.


Step 7 - Report average results per subject
For each subject (Science, English, Biology), calculate average results by reporting accuracy and QWK. Exclude Q10 because the grade level is different from others (aligning with the paper). Use Q1 and Q2 for “Science”, Q3, Q4, Q7, Q8, Q9 for “English”, Q5 and Q6 for “Biology”.

## Experiment Results


---
  
## 📖 Reference
