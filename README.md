# Topic: Predicting Startups Acquisition Status
The goal of this project is to predict a former startup’s acquisition status based on a company’s financial statistics.

### Objective:
The objective of the project is to predict whether a startup which will be Operating, IPO, Acquired or closed. This problem will be solved through a Supervised Machine Learning approach by training a model based on the history of startups that were either acquired or closed.

### Link to data:
 
  https://drive.google.com/file/d/1tWYkHYHm2HoiCajZ49Cs1K7sklWTdAbV/view?usp=sharing
       
### Summary:

The data contains industry trends, investment insights, and individual company information. Since the data was acquired on a trial basis, it only contains information about companies known. After training the model, we predict whether startups still operating, IPO, acquired, or closed.
 
### Dataset Overview
The dataset provides insights into various aspects of startups, including categories, status, founding date, country code, investment details, milestones, and more.

## Data Preprocessing

#### Removing Unnecessary Features:

Irrelevant and redundant columns were deleted as they do not directly contribute to predict acquisition status.  <br>

- Deleted redundant features: 'id', 'Unnamed: 0.1', 'entity_type', 'entity_id', 'parent_id', 'created_by', 'created_at', 'updated_at'.

- Deleted irrelevant features: 'domain', 'homepage_url', 'twitter_username', 'logo_url', 'logo_width', 'logo_height', 'short_description', 'description', 'overview', 'tag_list', 'name', 'normalized_name', 'permalink', 'invested_companies'.

#### Dealing with Missing Values
- The ROI column exhibits 99% missing values. Since it is the only financial column and its removal might impact predictions, we imputed the missing ROI values using the median.

- Several columns contain a high percentage of missing values. Features exceeding a 98% missing value threshold were dropped.
![image](https://github.com/rushikesh5555/Technocolab_MiniProject/assets/83490548/5260e786-eb17-42d0-bf6f-0ce05978cbe4)

- Calculated the new column "active_days" by taking the difference between the closed date and the founded date of the startup.

- Subsequently, we addressed missing values in crucial columns such as 'status', 'country_code', 'category_code', and 'founded_at'.
  
- Finally, missing values were imputed using mean, median, and mode.

#### Dealing with Outliers
- Outliers in the dataset, particularly in the 'active_days' and 'founded_at' columns, were addressed through the Interquartile Range (IQR) method. This step ensured a more robust analysis by removing extreme values.

#### Dealing with categorical features
Due to more than 30 unique categories in the 'category_code' and 'country_code' columns, using all categories could unnecessarily impact model performance. hence, kept the first 10 highly frequent categories as it is and labeled the remaining categories as 'other' class.  <br>
Transformed data columns ('founded_at', 'closed_at', 'first_funded_at', 'last_funding_at', 'first_milestone_at', 'last_milestone_at') into years and created a new variable, 'isClosed,' from 'closed_at' and 'status' columns.  <br>
## Feature Engineering

Applied the **Label Encoding** on the target column which is Status.  <br>
For the bivariate model, the status column is labeled as  <br>
Closed and acquired: 0  <br>
Operating and IPO: 1  <br>

For the multivariate model, the status column is labeled as   <br>
Closed: 0  <br>
Operating: 1  <br>
Ipo: 2  <br>
Acquired: 3  <br>

#### Creating new features
1. **Funding_usd_for_1_round**: Calculated by dividing funding_total_usd by total_funding_rounds, providing an insight into the average contribution of each funding round to the company.
2. **Milestone_diff**: Difference between the last milestone and the first milestone.

3. **Funding_year_diff**: Difference between the last funding year and the first funding year.

4. **Age_bucket**: Created and populated the "Age_bucket" column in the DataFrame, allowing for a categorical representation of the "active_days" variable.

### Scaling
**MinMax scaling** is used to transform the numerical data in a certain range, especially between 0 and 1.  <br>

### Feature Selection
Feature Selection is applied to reduce the input variable to our model by using only relevant data and getting rid of noise in data.  <br>
Employed the **mutual information score** criteria to identify and select the most relevant k features. This ensures a focused and efficient representation of the data for our model.

## Exploratory Data Analysis (EDA)

#### Feature Importance
During the analysis, we observed that certain features played a crucial role in determining the status of startups. Notably, the age of startups, funding trends, and category codes proved significant in understanding the dataset.

#### Status Distribution
![Status Distribution](https://github.com/rushikesh5555/Technocolab_MiniProject/assets/83490548/8072ba88-c03b-4466-83ae-b1bc59092d36)


The distribution of startup statuses revealed that operating and IPO startups dominate the dataset, with 1051 closed startups. This distribution is essential for understanding the balance of classes in our analysis.  <br>

#### Age and Funding Trends
![Age Bucket vs Status](https://github.com/rushikesh5555/Technocolab_MiniProject/assets/83490548/8e3643a1-96d0-47c6-9241-5c66b8a2ca44)


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

This application is designed to predict the acquisition status of startup companies based on selected input parameters.  <br>
It utilizes Streamlit to create an interactive interface allowing users to input various parameters, view feature-engineered data, and obtain predictions through two pipelines: one for binary classification and another for multi-class classification.

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
