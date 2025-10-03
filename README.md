# 🚗 Car Insurance Risk Factors — Logistic Regression Feature Benchmark

## 📖 Overview
This project analyzes a car insurance dataset to identify which **single feature** is the best predictor of whether a client will file an insurance claim.  
Instead of building a complex machine learning pipeline, the focus is on **logistic regression models for each feature individually**, ranked by accuracy. This makes the findings simple, explainable, and actionable for business stakeholders.

🔗 [Live Streamlit App](https://car-insurance-claims-5mids4f6bqvp6ubwwwwbri.streamlit.app/)

## ❓ Problem Statement
Insurance companies need to evaluate client risk quickly and accurately. Deploying complex machine learning models can be expensive, hard to monitor, and not always transparent.  
The business asked:  

**“If we had to choose only one feature to estimate claim likelihood, which feature would give the highest accuracy?”**

## 📊 About the Data
The dataset `car_insurance.csv` contains anonymized customer information, covering demographics, driving history, vehicle details, and claim outcomes.  

Key columns include:
- **Age group** (16–25, 26–39, 40–64, 65+)  
- **Gender** (Male/Female)  
- **Driving experience** (years, binned)  
- **Education** (none, high school, university)  
- **Income level** (poverty to upper class)  
- **Credit score** (0–1)  
- **Vehicle ownership/year/type**  
- **Marital status** and **children**  
- **Annual mileage**  
- **Violations, DUIs, past accidents**  
- **Outcome** (0 = no claim, 1 = claim)  

## ⚙️ Actions (Data Preparation)
1. **Removed duplicates** to avoid bias from repeated rows.  
2. **Handled missing data**:
   - `credit_score` → filled with mean value.  
   - `annual_mileage` → filled with mean value.  
3. **Dropped non-predictive identifiers** (`id`, `postal_code`).  
4. **Ran logistic regression per feature**:
   - Tried each feature as numeric.  
   - If failed, re-tried as categorical.  
   - Measured accuracy from the prediction table at a 0.5 threshold.  

## 📈 Results
- The best predictor of insurance claim outcome is:  

  **🏆 Driving Experience — Accuracy ≈ 0.77**  

- Other features (credit score, past accidents, vehicle type, etc.) showed lower predictive power.  
- A **ranking table and bar chart** are included in the Streamlit app to explore all features by accuracy.  

📷 **Results Preview**  
![Feature Ranking Table](images/ranking_table.png)  
*All features ranked by accuracy in a styled table.*  

![Accuracy Bar Chart](images/bar_chart.png)  
*Top features visualized for quick comparison.*  

## 🔍 Key Insights
- Customers with **less driving experience** are more likely to make a claim.  
- Demographics such as age and gender are less predictive compared to behavioral factors.  
- Simpler, interpretable features can still provide **strong baseline models** for risk assessment.  

## 📌 Business Impact
| Impact Area          | Value Delivered |
|----------------------|-----------------|
| **Risk Scoring**     | Provides a simple, interpretable feature (driving experience) that can be directly used for claim likelihood estimation. |
| **Operational Cost** | Avoids the complexity and cost of deploying black-box ML models by using a transparent single-feature logistic regression. |
| **Actionable Insight** | Empowers underwriters to adjust premium pricing strategies across driving experience groups. |
| **Decision Support** | Offers a clear benchmark for expanding toward multi-feature or advanced predictive models. |

## 📝 Takeaways
- **Driving Experience is the strongest predictor** of claim outcomes, with an accuracy of ~0.77.  
- Simpler models with interpretable features can deliver **practical business value quickly**.  
- **Future scope**: Combine top predictors to improve accuracy while balancing interpretability.  
- The approach serves as a **blueprint for explainable AI in insurance analytics**. 
