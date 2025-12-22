# Paper_Reproduce-Short_Answer_Scoring

## Overview
GPT_inference.ipynb was adapted from [lan-j/SAS_GPT4](https://github.com/lan-j/SAS_GPT4).

This repository reproduces a prompt-based short-answer scoring experiment using LLMs, following the methodology of the referenced paper. We sample student responses from the ASAP-SAS dataset, apply a fixed grading prompt to each response using Llama-3.1-405B Instruct on AWS Bedrock, and evaluate the model’s performance against human scores. The pipeline is organized into clear steps: data sampling (ensuring sufficient label coverage), model inference with predefined prompts, numeric score extraction, and evaluation using Accuracy and Quadratic Weighted Kappa (QWK). Results are reported per question, averaged across all questions, and further aggregated by subject (Science, English, Biology), excluding Q10 to match the paper’s grade-level alignment. 

We can understand the main findings by inspecting the final evaluation summaries, which show how closely the model’s scores agree with human graders under a simple prompt-based setup.

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

## Experiment Results

### Accuracy and QWK per Question
| Question | n_total | n_used | paper accuracy | llama 3.1 405b instruct accuracy | Δ accuracy | paper qwk | llama 3.1 405b instruct qwk | Δ qwk |
|----------|---------|--------|----------------|----------------------------------|------------|-----------|-------------------------------|-------|
| Q1 | 100 | 92 | 0.51 | 0.26 | -0.25 | 0.635 | 0.14 | -0.495 |
| Q2 | 100 | 97 | 0.53 | 0.31 | -0.22 | 0.654 | 0.25 | -0.404 |
| Q3 | 100 | 80 | 0.69 | 0.48 | -0.21 | 0.524 | 0.118 | -0.406 |
| Q4 | 100 | 84 | 0.706 | 0.46 | -0.246 | 0.518 | 0.198 | -0.320 |
| Q5 | 117 | 110 | 0.756 | 0.35 | -0.406 | 0.702 | 0.444 | -0.258 |
| Q6 | 112 | 108 | 0.781 | 0.46 | -0.321 | 0.737 | 0.42 | -0.317 |
| Q7 | 100 | 86 | 0.47 | 0.37 | -0.10 | 0.43 | 0.209 | -0.221 |
| Q8 | 100 | 91 | 0.47 | 0.33 | -0.14 | 0.49 | 0.18 | -0.310 |
| Q9 | 100 | 89 | 0.64 | 0.57 | -0.07 | 0.666 | 0.472 | -0.194 |
| Q10 | 100 | 92 | 0.73 | 0.46 | -0.27 | 0.753 | 0.206 | -0.547 |

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