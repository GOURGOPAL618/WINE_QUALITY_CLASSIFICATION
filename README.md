# 🍷 Wine Quality Classification System
# An Intelligent Machine Learning Web Application for Wine Quality Prediction Based on Physicochemical Properties

# 📋 Project Overview
Wine quality assessment has traditionally been performed by human experts through sensory evaluation — a process that is subjective, time-consuming, and expensive. This project addresses that challenge by building an intelligent Machine Learning Classification System that can automatically predict wine quality based on measurable physicochemical properties.
Using the WineQT dataset containing 1143 real wine samples with 11 chemical features, we trained and compared 4 different classification models (excluding Random Forest and Decision Trees) to find the best performing algorithm. The final system is deployed as an interactive Streamlit web application with live prediction, batch processing, and visual analytics.

# 👥 Team DYNAMO
## 👥 TEAM DYNAMO - Wine Quality Classification Project

| Name | Registration No. | Role | Responsibility |
|------|-----------------|------|----------------|
| **Gouragopal Mohapatra** | 459 | 👨‍💻 **Team Lead** | Project architecture, coordination, final review |
| **Arijit Kumar Mohanty** | 450 | 🤖 **ML Engineer** | Model building, training, evaluation |
| **Aayush Shroff** | 440 | 📊 **Data Analyst** | EDA, visualizations, feature engineering |
| **Ashok Kumar Nayak** | 452 | ⚙️ **Backend Dev** | Performance optimization, data pipeline |
| **Rudra Prasad Baral** | 436 | 🎨 **Frontend Dev** | UI/UX design, Streamlit interface |
| **Farhan Raja** | 458 | 📁 **Dataset Explorer** | Dataset exploration, preprocessing |

### 🎯 Team Contributions


🎯 Key Highlights

Dataset: WineQT.csv — 1143 samples, 11 physicochemical features
Problem Type: Multi-class Classification (3 quality levels)
Best Model: Support Vector Machine (SVM) with ~85% accuracy
No Random Forest or Decision Tree algorithms used
Deployment: Interactive Streamlit Web Application
Prediction Types: Single wine prediction + Batch CSV prediction
Export: Download predictions and performance reports as CSV


🍷 Quality Classification System
## 🍷 Wine Quality Classification Mapping

| Quality Score | Class Label | Class Code | Meaning |
|:-------------:|:-----------:|:----------:|---------|
| **3 – 4** | 🔴 **Low Quality** | `Class 0` | Poor wine, not recommended |
| **5 – 6** | 🟡 **Medium Quality** | `Class 1` | Average wine, acceptable |
| **7 – 8** | 🟢 **High Quality** | `Class 2` | Excellent wine, highly recommended |

### 📊 Class Distribution
- **Class 0 (Low)**: ~15% of dataset
- **Class 1 (Medium)**: ~70% of dataset
- **Class 2 (High)**: ~15% of dataset

### 🎯 Prediction Output Example

🤖 ML Models Used (No RF / No DT)
| Model | Type | Accuracy | F1 Score | Key Strength |
|-------|------|----------|----------|--------------|
| **SVM** | Kernel-based | 85% | 0.84 | Best overall performance |
| **KNN** | Similarity-based | 84% | 0.83 | Simple, effective |
| **Logistic Regression** | Linear | 82% | 0.81 | Fast, interpretable |
| **Naive Bayes** | Probabilistic | 80% | 0.79 | Lightweight, fast |

🏆 Winner: SVM with RBF Kernel — highest accuracy and F1 score across all quality classes.


# 📊 Dataset Information
Source: Wine Quality Dataset (UCI Machine Learning Repository)
File: WineQT.csv
Samples: 1143 wine records
Features: 11 physicochemical properties + 1 target variable
Feature Details
=====================================================================================================
| # | Tab Name          | What You Can Do                                                           |
=====================================================================================================
| 1 | 🔍 Data Explorer  | • View dataset preview (first 10 rows)                                   |
|   |                   | • Check dataset shape (rows, columns)                                    |
|   |                   | • See data types of all features                                         |
|   |                   | • Find missing values                                                    |
|   |                   | • Quick statistics summary                                               |
-----------------------------------------------------------------------------------------------------
| 2 | 📈 Visual Analytics| • Distribution histogram of wine quality                                 |
|   |                   | • Pie chart of quality classes (Low/Medium/High)                        |
|   |                   | • Quality class breakdown percentages                                    |
|   |                   | • Visual understanding of data balance                                   |
-----------------------------------------------------------------------------------------------------
| 3 | ⚡ Feature         | • Correlation heatmap between features                                   |
|   |   Engineering     | • Feature importance with quality                                        |
|   |                   | • Bar chart of correlations                                              |
|   |                   | • Identify most important features (alcohol, sulphates)                 |
-----------------------------------------------------------------------------------------------------
| 4 | 🤖 Model Training  | • Select ML models (LR/KNN/SVM/Naive Bayes)                             |
|   |                   | • Adjust test size (10-40%)                                              |
|   |                   | • Set random seed for reproducibility                                    |
|   |                   | • Click Train button to start training                                   |
|   |                   | • View training progress                                                 |
-----------------------------------------------------------------------------------------------------
| 5 | 📊 Performance     | • Accuracy bar chart comparison                                          |
|   |   Hub             | • Confusion matrix for each model                                        |
|   |                   | • Classification report (precision, recall, f1)                         |
|   |                   | • Download results as CSV                                                |
|   |                   | • Find best performing model                                             |
-----------------------------------------------------------------------------------------------------
| 6 | 🎯 Live Prediction | • Single Prediction:                                                     |
|   |                   |   - Enter 11 wine properties manually                                   |
|   |                   |   - Get quality prediction from all models                              |
|   |                   |   - View confidence scores                                               |
|   |                   | • Batch Prediction:                                                      |
|   |                   |   - Upload CSV with multiple wines                                       |
|   |                   |   - Process 100+ wines at once                                           |
|   |                   |   - Download results with predictions                                    |
=====================================================================================================
📁 Project Structure
wine-quality-classifier/
│
├── app.py                    
├── WineQT.csv                
├── requirements.txt          
├── README.md                 
├── Wine_Quality_Classification.ipynb             
└── License                   

🚀 Installation & Setup Guide
Step 1: Clone the Repository
bashgit clone https://github.com/yourusername/wine-quality-classifier.git
cd wine-quality-classifier
Step 2: Install Required Dependencies
bashpip install -r requirements.txt
Step 3: Add the Dataset
Download WineQT.csv and place it in the root project folder.
Step 4: Launch the Application
bashstreamlit run app.py
```
The app will open automatically at `http://localhost:8501`

---

## 📦 Requirements
```
streamlit==1.28.0
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
plotly==5.17.0
matplotlib==3.7.0
seaborn==0.12.0

💻 Application Features — 6 Interactive Tabs
#	Tab Name	           What You Can Do
1	🔍 Data Exploration	• View dataset preview (first 10 rows)
                                • Check dataset shape (rows, columns)
                                • See data types of all features
                                • Find missing values
                                • Quick statistics summary

2	📈 Visual Analytics	• Distribution histogram of wine quality
                                • Pie chart of quality classes (Low/Medium/High)
                                • Quality class breakdown percentages
                                • Visual understanding of data balance

3	⚡ Feature Engineering	• Correlation heatmap between features
                                • Feature importance with quality
                                • Bar chart of correlations
                                • Identify most important features (alcohol, sulphates)

4	🤖 Model Training	• Select ML models (LR/KNN/SVM/Naive Bayes)
                                • Adjust test size (10-40%)
                                • Set random seed for reproducibility
                                • Click Train button to start training
                                • View training progress

5	📊 Performance Hub	• Accuracy bar chart comparison
                                • Confusion matrix for each model
                                • Classification report (precision, recall, f1)
                                • Find best performing model

6	🎯 Live Prediction	• Single Prediction:
                                - Enter 11 wine properties manually
                                - Get quality prediction from all models
                                - View confidence scores
                                • Batch Prediction:
                                - Upload CSV with multiple wines
                                - Process 100+ wines at once
                                - Download results with predictions

🎨 UI Design Features
Header Section

Gradient background with wine theme
Project title and description
Feature badges (dataset size, model count, accuracy)

Sidebar

Dataset upload button
Model selection checkboxes
Train/test split slider
Team DYNAMO members with registration numbers
Live dataset statistics

Main Dashboard

6 interactive tabs with smooth navigation
Metric cards showing key statistics
Plotly interactive visualizations
Real-time prediction results with confidence scores

Footer

Glowing gradient separator line
All team members with registration numbers
Project year 2026



## 📈 Sample Prediction Results

**Single Wine Prediction**
```
Input Features:
  Alcohol        : 12.5%
  pH             : 3.2
  Volatile Acidity: 0.45
  Sulphates      : 0.65
  Density        : 0.997

Output:
  SVM  → High Quality (Class 2) | Confidence: 85%
  KNN  → High Quality (Class 2) | Confidence: 82%
  LR   → Medium Quality (Class 1) | Confidence: 78%
```

**Batch Prediction**
```
100 wines processed in under 2 seconds
Output: predictions.csv with quality class for each wine
```

---

## 🐛 Troubleshooting Guide

| Problem | Cause | Solution |
|---------|-------|----------|
| `EmptyDataError` | Wrong CSV format | Ensure WineQT.csv has correct headers and no empty rows |
| Mixed numeric types | String values in numeric columns | All values are auto-converted to float on load |
| Models not training | Missing columns | Verify all 11 feature columns are present in dataset |
| App not launching | Streamlit not installed | Run `pip install streamlit==1.28.0` |
| Low accuracy | Data not scaled | StandardScaler is applied automatically before training |

---

## 🎯 Real-World Use Cases

**🏭 Wine Industry**
- Automated quality control in production lines
- Reduce dependency on human tasters
- Lower production costs through early detection of poor batches

**🍷 Wine Tasting & Certification**
- Expert verification and second opinion tool
- Consistency checking across different batches
- Training tool for new wine quality assessors

**📦 Export & Trading Business**
- Automated quality grading for large shipments
- Price determination based on quality class
- Market segmentation and targeting

---

## 🔄 Future Enhancements

| Feature | Description | Priority |
|---------|-------------|----------|
| Deep Learning Model | Neural network for higher accuracy | High |
| Real-time Data Updates | Connect to live sensor data | Medium |
| Mobile Responsive Design | Better UI on phones/tablets | Medium |
| REST API Endpoint | Allow external apps to call predictions | High |
| Model Retraining | Retrain on new uploaded data | Medium |
| More Visualizations | 3D plots, scatter matrices | Low |

---

## 👨‍💻 Team Contributions

| Name | Reg. No. | Key Contributions |
|------|----------|------------------|
| **Gouragopal Mohapatra** | 459 | 👨‍💼 **Team Lead** - Overall project architecture, team coordination, final integration and review |
| **Arijit Kumar Mohanty** | 450 | 🤖 ML model development, hyperparameter tuning, model evaluation and comparison |
| **Aayush Shroff** | 440 | 📊 Exploratory data analysis, all visualizations, feature engineering |
| **Ashok Kumar Nayak** | 452 | ⚙️ Backend logic, data pipeline, performance optimization, export functionality |
| **Rudra Prasad Baral** | 436 | 🎨 Streamlit UI design, frontend layout, tab structure, user experience |
| **Farhan Raja** | 458 | 📁 Dataset exploration and Feature Selection |

### 🎯 Project Overview


---

## 🎓 Final Project Summary
```
╔══════════════════════════════════════════════════╗
║         WINE QUALITY CLASSIFICATION              ║
║              TEAM DYNAMO — 2026                  ║
╠══════════════════════════════════════════════════╣
║  Project Type  : Machine Learning Classification ║
║  Dataset       : WineQT (1143 samples)           ║
║  Features      : 11 physicochemical properties   ║
║  Classes       : 3 (Low, Medium, High)           ║
║  Models Used   : LR, KNN, SVM, Naive Bayes       ║
║  Best Model    : SVM (~85% accuracy)             ║
║  Deployment    : Streamlit Web Application       ║
║  Team Size     : 5 Members                       ║
║  Course        : AI & Data Science — 2026        ║
╚══════════════════════════════════════════════════╝

📄 License
This project is created purely for educational purposes as part of the AI & Data Science Course (2026). Dataset credits go to the UCI Machine Learning Repository.

🙏 Acknowledgments

Dataset: Wine Quality Dataset — UCI Machine Learning Repository
ML Framework: scikit-learn
Web Framework: Streamlit
Visualization: Plotly, Matplotlib, Seaborn
Guidance: Course Instructors & Mentors


Made with 🩵 by TEAM DYNAMO | AI & Data Science Project 2026
Arijit (450) • Gouragopal (459) • Aayush (440) • Ashok (452) • Rudra (436) • Farhan (458)
