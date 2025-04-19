# Hybrid Trust Model for Cloud Service Provider (CSP) Selection  
**A bidirectional trust model integrating NLP-based requirement extraction and malicious feedback filtering for reliable CSP recommendations.**  

![GitHub](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.7%2B-green)  

## 📝 Overview  
This project proposes a **hybrid trust model** to address the challenges of Cloud Service Provider (CSP) selection by:  
1. **Bidirectional Trust Evaluation**: Combines CSP trust scores (based on QoS parameters) and user feedback trust (collaborative filtering).  
2. **NLP-Powered Requirement Extraction**: Allows users to input requirements via text or numerical weights using a rule-based NLP module (*98% accuracy*).  
3. **Malicious Feedback Filtering**: Detects fake reviews using multivariate outlier detection (IQR method) without requiring frequent feedback updates.  

**Key Result**: Outperforms existing models by **59.01%** (vs. QoS-only) and **17.68%** (vs. feedback-only).  

---

## 🛠️ Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/BhanuPrakashNani/hybrid-trust-model.git  
   cd hybrid-trust-model  
   ```  
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  # Includes spaCy, pandas, scikit-learn, etc.  
   python -m spacy download en_core_web_sm  # NLP model  
   ```  

---

## 🚀 Usage  
### 1. **Trust Calculation**  
Run the hybrid trust model:  
```python  
python src/trust_calculation.py \  
    --user_requirements "uptime: high, downtime: very low" \  
    --num_recommendations 5  
```  
*Input formats*: Text (e.g., `"uptime: high"`) or numerical weights (e.g., `{"uptime": 4.5}`).  

### 2. **Malicious Feedback Detection**  
Filter outliers from feedback data:  
```python  
python src/feedback_filter.py --feedback_csv data/feedback.csv  
```  

### 3. **NLP Weight Extractor**  
Test the rule-based NLP module:  
```python  
python src/nlp_extractor.py --text "I need very high uptime and low downtime"  
```  
*Output*: `{"uptime": 5, "downtime": 1}`  

---

## 📂 Repository Structure  
```  
├── data/                    # Sample datasets  
│   ├── csp_parameters.csv   # CSP QoS promises  
│   └── feedback.csv         # User feedback logs  
├── src/  
│   ├── trust_calculation.py # Hybrid trust algorithm  
│   ├── feedback_filter.py   # Outlier detection  
│   └── nlp_extractor.py     # NLP weight extraction  
├── docs/  
│   └── ICCCNT_2021_Paper.pdf # Published paper  
└── requirements.txt         # Dependencies  
```  

---

## 📊 Results  
- **Optimal Weights**: `0.3` (QoS trust) + `0.7` (feedback trust).  
- **Accuracy**: 70.2% successful recommendations (vs. 26.4% for QoS-only).  
- **Malicious Feedback**: 0.3% improvement in recommendations after filtering.

---

## 📜 Citation  
If you use this work, please cite:  
```bibtex  
@inproceedings{poluparthi2021hybrid,  
  title={A Hybrid Trust Model for Cloud Service Provider Selection with NLP Support and Malicious User Feedback Filtering},  
  author={Poluparthi, Bhanu Prakash and Kishan, G. Mani and Praneeth, V. Bala Sai and Manikanta, A. and Sarath, Greeshma},  
  booktitle={2021 12th International Conference on Computing Communication and Networking Technologies (ICCCNT)},  
  year={2021},  
  organization={IEEE}  
}  
```  

---

## 🤝 Contributing  
Contributions are welcome! Open an issue or submit a PR for:  
- Expanding QoS parameters.  
- Enhancing NLP rule-based models.  
- Optimizing outlier detection.  

---

## 📄 License  
MIT © [Bhanu Prakash Poluparthi](https://linkedin.com/in/bhanu-prakash-poluparthi)  

--- 
