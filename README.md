## Feature Engineering

Applied the **Label Encoding** on the target column which is Status.
For the bivariate model status column is labeled as  <br>
Closed and acquired: 0  <br>
Operating and ipo: 1  <br>

For the multivariate model status column is labeled as 
Closed: 0<be>
Operating: 1<be>
Ipo: 2<be>
Acquired: 3<be>

#### Creating new features
1. **Funding_usd_for_1_round**: Calculated by dividing funding_total_usd by total_funding_rounds, providing an insight into the average contribution of each funding round to the company.
2. **Milestone_diff**: Difference between the last milestone and the first milestone.

3. **Funding_year_diff**: Difference between the last funding year and the first funding year.

4. **Age_bucket**: Created and populated the "Age_bucket" column in the DataFrame, allowing for a categorical representation of the "active_days" variable.

### Scaling
**MinMax scaling** is used to transform the numerical data in a certain range, especially between 0 and 1.

### Feature Selection
Feature Selection is applied to reduce the input variable to our model by using only relevant data and getting rid of noise in data.
Employed the **mutual information score** criteria to identify and select the most relevant k features. This ensures a focused and efficient representation of the data for our model.


## Exploratory Data Analysis (EDA)

#### Feature Importance
During the analysis, we observed that certain features played a crucial role in determining the status of startups. Notably, the age of startups, funding trends, and category codes proved significant in understanding the dataset.

#### Status Distribution
![Status Distribution](D:\Mohamed Sheriff\Projects\Technocolabs Machine Learning Internship\Startup-Acquisition-Using-Crunchbase\Imgs/Status Distribution.png)

The distribution of startup statuses revealed that operating and IPO startups dominate the dataset, with 1051 closed startups. This distribution is essential for understanding the balance of classes in our analysis.

#### Age and Funding Trends
![Age Bucket vs. Status](D:\Mohamed Sheriff\Projects\Technocolabs Machine Learning Internship\Startup-Acquisition-Using-Crunchbase\Imgs/Age Bucket vs Status.png)

Startups with an age range above 2500 days demonstrated a higher probability of being in operation. Additionally, funding trends were explored across different age groups, providing insights into the dynamics of startup funding over time.

#### Category Code Analysis
The dataset includes startups from various categories. Notably, the biotech category stood out, receiving almost double the funding compared to enterprise and hardware startups.

#### Geographical and Funding Insights
A Pareto chart highlighted that startups from the USA, Canada, and Germany receive higher funding than other countries. This geographical insight could have strategic implications for investors and policymakers.

#### Conclusion and Next Steps
The EDA phase provided valuable insights into the startup dataset. Key findings include age-related trends, geographical funding patterns, and the influence of category codes on funding. The identified patterns and outliers will inform subsequent steps in the project, including feature engineering, modeling, and potential further data collection.


## Model Building

#### Metrics considered for Model Evaluation:
Accuracy, Precision, sensitivity, specificity

- Accuracy: It is the ratio of overall correctly predicted instances to the total number of instances in the dataset.
- Precision: precision is the measure that represents the correctly predicted positive classes from the total predicted positive classes.
- Sensitivity: sensitivity represents the correctly predicted positive class from the total number of actual positive instances.
- Specificity: specificity represents the correctly predicted Negative class from the total number of actual negative instances.

### Bivariate Model

- The 'status' column is selected as the target column, and the expected statuses are 'Operating' and 'Closed'.
- Feature selection on the training data is done using the SelectKBest method from scikit-learn. We selected 5 features namely 'founded_at',
 'ROI','active_days','Age_group_2500-4000'and'Age_group_4000-5500'

#### Regularised Logistic Regression

- Regularized Logistic Regression (RLR) was chosen as a modeling algorithm due to its effectiveness in binary classification tasks.
- The model underwent hyperparameter tuning using techniques like GridSearchCV, ensuring optimal performance by fine-tuning parameters such as penalty, solver, and regularization strength (C).
- Generated ROC curves to assess the performance of the model. The resulting AUC was found to be 92%.
- Derived the threshold probability as 0.9 from the overlapping criteria of accuracy, sensitivity, and specificity.
- Achieving a remarkable *98% precision*, 79% accuracy, 77% sensitivity on the testing dataset.

#### Random Forest Classifier

- The Random Forest algorithm was employed for its ability to handle complex relationships and feature interactions
- The Random Forest model underwent extensive hyperparameter tuning using GridSearchCV, optimizing parameters like the number of trees,  maximum depth, and minimum samples split to enhance predictive accuracy.
- Performance metrics, including accuracy, precision, recall, and specificity, were thoroughly evaluated on both training and testing datasets.
- The Random Forest model demonstrated robustness, achieving 96% accuracy on training and 95% on testing datasets. 

### Multivariate Model

- For the multivariate model we are taking KNN and XGBoost classification Algorithms.
- Used the whole dataset to train the models, the 'status' column is selected as the target column, and the expected statuses are 'Operating', 'Closed', 'Acquired', and 'IPO'.
- Feature selection on the training data is done using the SelectKBest method from scikit-learn. We selected 10 features namely 'founded_at',
'funding_total_usd','ROI','active_days','category_code_mobile','country_code_IRL','funding_usd_for_1_round','Age_group_2500-4000','Age_group_5500-7000'and'Age_group_7000-10000'
 
#### K-Nearest Neighbors 
 
- Employed the K-Nearest Neighbors (KNN) algorithm for its simplicity and effectiveness in multiclass classification scenarios.
- Utilized GridSearchCV to fine-tune the model's hyperparameters, optimizing key parameters like the number of neighbors, distance metric, and weighting scheme. The best configuration was found to have 6 neighbors, using the Manhattan distance metric and uniform weighting.
- Achieved a training accuracy of 80%, with a precision of 53% and recall of 42%. The model exhibited a mean absolute error of 0.3 and a root mean squared error of 0.72, showcasing its ability to capture patterns in the training dataset.
- On the testing dataset, the model demonstrated a test accuracy of 75%, with a precision of 39% and recall of 39%. The mean absolute error was 0.38, and the root mean squared error was 0.8, indicating reasonable generalization to new, unseen data.
 
#### XGBoost Multiclass Classification
 
- Conducted a hyperparameter search for XGBoost, fine-tuning critical parameters like learning rate, number of trees, and tree depth, resulting in an optimized configuration.
- Achieved a peak accuracy of 91.5% on the training dataset using the best set of hyperparameters, ensuring robust performance during model training.
- Evaluated the XGBoost model on the training dataset, showcasing strong metrics with *92% accuracy*, 72% precision, and 55% recall, demonstrating its effectiveness in multiclass classification.
- Demonstrated the model's ability to generalize on unseen data, maintaining a high accuracy of 89% on the testing dataset, with balanced precision and recall metrics.

## Pipeline Construction

- Pipeline encompasses key steps, including MinMax scaling, optimal feature selection using mutual information score, and model fitting.
- Two distinct pipelines were formulated for bivariate and multivariate models.
- The optimal bivariate model, Regularized Logistic Regression, stood out with the highest precision. Simultaneously, the multivariate model, XGBoost, was chosen for its remarkable accuracy.

## Deployment Stage

This application is designed to predict the acquisition status of startup companies based on selected input parameters. It utilizes Streamlit to create an interactive interface allowing users to input various parameters, view feature-engineered data, and obtain predictions through two pipelines: one for binary classification and another for multi-class classification.

### Features

- **Interactive User Interface:** Users can select input parameters through sliders and a select box in the sidebar.

- **Data Display:** The chosen parameters are displayed in a table, showcasing feature-engineered data for user review.

- **Prediction:**
  - _Binary Classification:_ Uses a regularized Logistic Regression Model to predict whether a startup will be "Open" or "Closed".
  - _Multiclass Classification:_ Utilizes XGBoost Classifier Model to predict whether a startup will fall into one of four categories: "Operating", "IPO", "Closed", or "Acquired".

### How to use

- **Running the App:** Execute the following command to run the Streamlit application.
  `streamlit run app.py`

- **Using the Application:** Once the app is running, access it through the provided local or hosted URL. Adjust the input parameters using the sidebar sliders and select box. Observe the displayed table showcasing the selected parameters and the corresponding feature-engineered data. Obtain predictions for acquisition status from both pipelines.
