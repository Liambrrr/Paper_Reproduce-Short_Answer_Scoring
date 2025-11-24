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

### Accuracy and QWK per Question
| Question | n_total | n_used | paper accuracy | llama 3.1 405b instruct accuracy | paper qwk | llama 3.1 405b instruct qwk |
|----------|---------|--------|----------------|----------------------------------|-----------|-------------------------------|
| Q1       | 100     | 92     | 0.51           | 0.26                             | 0.635     | 0.14                          |
| Q2       | 100     | 97     | 0.53           | 0.31                             | 0.654     | 0.25                          |
| Q3       | 100     | 80     | 0.69           | 0.48                             | 0.524     | 0.118                         |
| Q4       | 100     | 84     | 0.706          | 0.46                             | 0.518     | 0.198                         |
| Q5       | 117     | 110    | 0.756          | 0.35                             | 0.702     | 0.444                         |
| Q6       | 112     | 108    | 0.781          | 0.46                             | 0.737     | 0.42                          |
| Q7       | 100     | 86     | 0.47           | 0.37                             | 0.43      | 0.209                         |
| Q8       | 100     | 91     | 0.47           | 0.33                             | 0.49      | 0.18                          |
| Q9       | 100     | 89     | 0.64           | 0.57                             | 0.666     | 0.472                         |
| Q10      | 100     | 92     | 0.73           | 0.46                             | 0.753     | 0.206                         |

### Average Accuracy and QWK
| paper_avg_accuracy | replication_avg_accuracy | paper_avg_qwk | replication_avg_qwk |
|--------------------|---------------------------|---------------|-----------------------|
| 0.63               | 0.41                      | 0.611         | 0.264                 |

### Accuracy and QWK per Subject
| Subject | Questions                  | Avg Accuracy | Avg QWK |
|---------|-----------------------------|--------------|---------|
| Science | Q1, Q2                      | 0.29         | 0.195   |
| English | Q3, Q4, Q7, Q8, Q9          | 0.44         | 0.235   |
| Biology | Q5, Q6                      | 0.41         | 0.432   |

---
  
## 📖 Reference
Lan Jiang and Nigel Bosch. 2024. Short answer scoring with GPT-4. In Proceedings of the Eleventh ACM Conference on Learning @ Scale (L@S '24). Association for Computing Machinery, New York, NY, USA, 438–442. https://doi.org/10.1145/3657604.3664685