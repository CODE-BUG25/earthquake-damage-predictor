#  Earthquake Damage Prediction using Machine Learning

<div align="center">
  
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-4C72B0?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-Website-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-Styling-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-Interactive%20Website-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Deployed-222222?style=for-the-badge&logo=githubpages&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-End--to--End-blue?style=for-the-badge)
![EDA](https://img.shields.io/badge/EDA-Exploratory%20Data%20Analysis-8E44AD?style=for-the-badge)
![Hyperparameter Tuning](https://img.shields.io/badge/Hyperparameter-Tuning-success?style=for-the-badge)
![Git](https://img.shields.io/badge/Git-Version%20Control-F05032?style=for-the-badge&logo=git&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

###  Predict Earthquake Damage Severity with Machine Learning

**An end-to-end Machine Learning project covering Data Cleaning, EDA, Feature Engineering, Model Training, Hyperparameter Tuning, Evaluation, and Deployment using Streamlit.**

</div>

---

#  Project Overview

Earthquakes are among the most destructive natural disasters, often causing extensive damage to infrastructure and human life. Early estimation of the expected damage level helps governments, emergency responders, and disaster management agencies allocate resources more effectively.

This project leverages Machine Learning to predict the expected damage severity of an earthquake using historical earthquake characteristics such as magnitude, depth, intensity, tsunami information, significance, and engineered risk features.

The project demonstrates the complete Machine Learning lifecycle—from raw data preprocessing to deployment through an interactive web application.

---

#  Objectives

- Build an end-to-end Machine Learning pipeline
- Perform extensive Exploratory Data Analysis (EDA)
- Engineer meaningful features
- Compare multiple Machine Learning algorithms
- Optimize models using Hyperparameter Tuning
- Deploy the best model using Interactive HTML Website
- Create a professional, production-style ML project

---

#  Features

✔ Data Cleaning

✔ Missing Value Handling

✔ Duplicate Removal

✔ Exploratory Data Analysis (EDA)

✔ Feature Engineering

✔ Feature Scaling

✔ Categorical Encoding

✔ Multiple Machine Learning Algorithms

✔ Cross Validation

✔ Hyperparameter Tuning

✔ Feature Importance Analysis

✔ Model Evaluation

✔Interactive HTML Website

✔ Interactive Dashboard

✔ Download Predictions

✔ GitHub Ready Project Structure

---

#  Project Structure

```text
Earthquake-Damage-Prediction/

│
├── data/
│   ├── earthquake_1995-2023.csv
│   ├── earthquake_cleaned.csv
│   ├── processed_dataset.csv
│   └── model_results.csv
│
├── models/
│   ├── final_earthquake_model.pkl
│   ├── tuned_model.pkl
│   ├── preprocessor.pkl
│   └── label_encoder.pkl
│
├── notebooks/
│   ├── 01_Data_Cleaning_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Building.ipynb
│   ├── 04_Model_Comparison_Hyperparameter_Tuning.ipynb
│   └── 05_Model_Evaluation_Deployment.ipynb
│
├── Interactive HTML Website
├── requirements.txt
├── README.md
└── images/
```

---

#  Machine Learning Workflow

```text
Historical Earthquake Data
              │
              ▼
      Data Cleaning
              │
              ▼
 Exploratory Data Analysis
              │
              ▼
   Feature Engineering
              │
              ▼
 Encoding & Scaling
              │
              ▼
 Train-Test Split
              │
              ▼
 Multiple ML Models
              │
              ▼
 Hyperparameter Tuning
              │
              ▼
 Best Model Selection
              │
              ▼
   Model Evaluation
              │
              ▼
  Interactive HTML Website
```

---

#  Dataset Features

The project uses earthquake-related attributes such as:

- Magnitude
- Depth
- CDI (Community Determined Intensity)
- MMI (Modified Mercalli Intensity)
- Significance Score
- Tsunami Indicator
- Alert Level
- Latitude
- Longitude
- Country
- Continent
- Date & Time

### Engineered Features

- Magnitude Category
- Depth Category
- Energy Index
- Risk Score
- High Alert Flag
- Tsunami Flag
- Years Since Event

---

#  Machine Learning Models

The project compares multiple classification algorithms:

| Model | Status |
|--------|--------|
| Logistic Regression | ✅ |
| Decision Tree | ✅ |
| Random Forest | ✅ |
| Gradient Boosting | ✅ |
| Extra Trees | ✅ |
| AdaBoost | ✅ |
| K-Nearest Neighbors | ✅ |
| Support Vector Machine | ✅ |
| Naive Bayes | ✅ |

---

#  Model Evaluation

The project evaluates models using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Cross Validation
- Feature Importance
- Learning Curve
- Validation Curve

---

#  Application Screenshots

###  Home Page

![Home Page](assets/demo/home_page.png)

---

###  Prediction Page

![Prediction Page](assets/demo/prediction_page.png)

---

###  Model Information

![Model Information](assets/demo/model_info.png)

---

#  Data Visualizations

### Earthquakes Per Year

![Earthquakes Per Year](assets/01_visualizations/Earthquakes%20Per%20Year.png)

### Damage Level Count

![Damage Level Count](assets/01_visualizations/Damage%20Level%20Count.png)

### Magnitude Distribution

![Magnitude Distribution Plot](assets/01_visualizations/Magnitude%20Distribution%20Plot.png)
### Magnitude vs Depth

![Magnitude vs Depth Scatterplot](assets/01_visualizations/Magnitude%20vs%20Depth%20Scatterplot.png)

### Correlation Heatmap

![Correlation Heatmap](assets/01_visualizations/Correlation%20Heatmap.png)

---

#  Model Comparison

### Accuracy of Machine Learning Models

![Accuracy of Machine Learning Models](assets/02_model_comparison/Accuracy%20of%20Machine%20Learning%20Models.png)

### Cross Validation Accuracy

![Cross Validation Accuracy of Machine Learning Models](assets/02_model_comparison/Cross%20Validation%20Accuracy%20of%20Machine%20Learning%20Models.png)

### F1 Score

![F1 Score of Machine Learning Models](assets/02_model_comparison/F1%20Score%20of%20Machine%20Learning%20Models.png)

---

#  Model Evaluation

### Confusion Matrix

![Confusion Heatmap of Actual vs Predicted](assets/03_evaluation/Confusion%20heatmap%20of%20Actual%20vs%20Predicted.png)
---

#  Feature Analysis

### Feature Importance

![Feature Importance](assets/04_feature_analysis/Feature%20Importance.png)


---

# ⚙️ Hyperparameter Tuning

### Learning Curve

![Learning Curve](assets/05_tuning/Learning%20Curve.png)

### Validation Curve

![Validation Curve (Random Forest)](assets/05_tuning/Validation%20Curve%20(Random%20Forest).png)
---

#  Installation

### Clone the Repository

```bash
git clone https://github.com/CODE-BUG25/Earthquake-Damage-Prediction.git
```

### Navigate to the Project Folder

```bash
cd Earthquake-Damage-Prediction
```

### Install Required Libraries

```bash
pip install -r requirements.txt
```

### Launch the Website

Open the `website/index.html` file in your preferred web browser.

---

#  Required Libraries

```
Python
NumPy
Pandas
Matplotlib
Seaborn
Plotly
Scikit-Learn
Joblib
Pickle
```

---

#  Future Improvements

- Integrate real-world damage labels
- Add XGBoost, LightGBM & CatBoost
- Explain predictions using SHAP
- Real-time earthquake data integration
- REST API deployment
- Docker support
- Cloud deployment (AWS/Azure/GCP)

---

#  Learning Outcomes

This project demonstrates practical experience with:

- Data Preprocessing
- Feature Engineering
- Exploratory Data Analysis
- Machine Learning
- Hyperparameter Optimization
- Model Evaluation
- Model Deployment
- Interactive Dashboard Development

---

#  License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this project under the terms of the MIT License.

See the **LICENSE** file for more details.

---

#  Author

## Sourabh Verma

**B.Tech Computer Science Student**

Passionate about:

- Artificial Intelligence
- Machine Learning
- Data Science
- Deep Learning
- Open Source
- Space Technology

### Connect with me

GitHub

https://github.com/CODE-BUG25

LinkedIn

https://www.linkedin.com/in/sourabh-verma-0b480036b/

Email

sverma25072006@gmail.com

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a Star!


<div align="center">

## 🚀 Happy Coding!

### ⭐ Star • 🍴 Fork • 🛠️ Contribute • 📢 Share

**Made with ❤️ by Sourabh Verma**

</div>
```
