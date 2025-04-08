# handson-10-MachineLearning-with-MLlib.

#  Customer Churn Prediction with MLlib

This project uses Apache Spark MLlib to predict customer churn based on structured customer data. You will preprocess data, train classification models, perform feature selection, and tune hyperparameters using cross-validation.

---



Build and compare machine learning models using PySpark to predict whether a customer will churn based on their service usage and subscription features.

---

##  Dataset

The dataset used is `customer_churn.csv`, which includes features like:

- `gender`, `SeniorCitizen`, `tenure`, `PhoneService`, `InternetService`, `MonthlyCharges`, `TotalCharges`, `Churn` (label), etc.

---

##  Tasks

### Task 1: Data Preprocessing and Feature Engineering

**Objective:**  
Clean the dataset and prepare features for ML algorithms.

**Steps:**
1. Fill missing values in `TotalCharges` with 0.
2. Encode categorical features using `StringIndexer` and `OneHotEncoder`.
3. Assemble numeric and encoded features into a single feature vector with `VectorAssembler`.

**Code Output:**

```
+--------------------+-----------+
|features            |ChurnIndex |
+--------------------+-----------+
|[0.0,12.0,29.85,29...|0.0        |
|[0.0,1.0,56.95,56....|1.0        |
|[1.0,5.0,53.85,108...|0.0        |
|[0.0,2.0,42.30,184...|1.0        |
|[0.0,8.0,70.70,151...|0.0        |
+--------------------+-----------+
```
---

### Task 2: Train and Evaluate Logistic Regression Model

**Objective:**  
Train a logistic regression model and evaluate it using AUC (Area Under ROC Curve).

**Steps:**
1. Split dataset into training and test sets (80/20).
2. Train a logistic regression model.
3. Use `BinaryClassificationEvaluator` to evaluate.

**Code Output Example:**
```
Logistic Regression Model Accuracy: 0.83
```

---

###  Task 3: Feature Selection using Chi-Square Test

**Objective:**  
Select the top 5 most important features using Chi-Square feature selection.

**Steps:**
1. Use `ChiSqSelector` to rank and select top 5 features.
2. Print the selected feature vectors.

**Code Output Example:**
```
+--------------------+-----------+
|selectedFeatures    |ChurnIndex |
+--------------------+-----------+
|[0.0,29.85,0.0,0.0...|0.0        |
|[1.0,56.95,1.0,0.0...|1.0        |
|[0.0,53.85,0.0,1.0...|0.0        |
|[1.0,42.30,0.0,0.0...|1.0        |
|[0.0,70.70,0.0,1.0...|0.0        |
+--------------------+-----------+

```

---

### Task 4: Hyperparameter Tuning and Model Comparison

**Objective:**  
Use CrossValidator to tune models and compare their AUC performance.

**Models Used:**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosted Trees (GBT)

**Steps:**
1. Define models and parameter grids.
2. Use `CrossValidator` for 5-fold cross-validation.
3. Evaluate and print best model results.

**Code Output Example:**
```
Tuning LogisticRegression...
LogisticRegression Best Model Accuracy (AUC): 0.84
Best Params for LogisticRegression: regParam=0.01, maxIter=20

Tuning DecisionTree...
DecisionTree Best Model Accuracy (AUC): 0.77
Best Params for DecisionTree: maxDepth=10

Tuning RandomForest...
RandomForest Best Model Accuracy (AUC): 0.86
Best Params for RandomForest: maxDepth=15
numTrees=50

Tuning GBT...
GBT Best Model Accuracy (AUC): 0.88
Best Params for GBT: maxDepth=10
maxIter=20

```
---

##  Execution Instructions

### 1. Prerequisites

- Apache Spark installed
- Python environment with `pyspark` installed
- `customer_churn.csv` placed in the project directory

### 2. Run the Project

### 2. Run the Pr

```bash
spark-submit churn_prediction.py
```
### Make sure to include your original ouput and explain the code



This project demonstrates a complete machine learning pipeline for predicting customer churn using PySpark's MLlib. The following tasks are performed:

1. **Data Preprocessing and Feature Engineering**
2. **Building and Training a Logistic Regression Model**
3. **Feature Selection Using Chi-Square Test**
4. **Hyperparameter Tuning and Model Comparison**

---

### Requirements

Before you start, ensure that you have the following installed:
- Python 3.x
- Apache Spark (PySpark)
- Java 8 or higher
- `pyspark` library

You can install the required Python libraries using the following command:

```bash
pip install pyspark
```

### Task 1: Data Preprocessing and Feature Engineering

In this task, the dataset is preprocessed to handle missing values, encode categorical features, and assemble them into a feature vector for machine learning models.

#### Steps:
1. **Load Data**: The dataset is read from a CSV file, and the schema is inferred.
2. **Fill Missing Values**: Missing values in the `TotalCharges` column are replaced with 0.
3. **Index Categorical Features**: Using `StringIndexer`, the categorical variables (`gender`, `PhoneService`, `InternetService`, and `Churn`) are converted into numerical indexes.
4. **One-Hot Encoding**: The indexed categorical features are then one-hot encoded using `OneHotEncoder`.
5. **Feature Assembly**: The features are combined into a single feature vector using `VectorAssembler`.

#### Code:
```python
def preprocess_data(df):
    # Fill missing values
    df = df.fillna({'TotalCharges': 0})

    # Encode categorical variables
    gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_index")
    phone_service_indexer = StringIndexer(inputCol="PhoneService", outputCol="phone_service_index")
    internet_service_indexer = StringIndexer(inputCol="InternetService", outputCol="internet_service_index")
    churn_indexer = StringIndexer(inputCol="Churn", outputCol="ChurnIndexed")
    
    gender_encoder = OneHotEncoder(inputCol="gender_index", outputCol="gender_onehot")
    phone_service_encoder = OneHotEncoder(inputCol="phone_service_index", outputCol="phone_service_onehot")
    internet_service_encoder = OneHotEncoder(inputCol="internet_service_index", outputCol="internet_service_onehot")

    assembler = VectorAssembler(inputCols=["gender_onehot", "phone_service_onehot", "internet_service_onehot", 
                                           "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"],
                                outputCol="features")

    pipeline = Pipeline(stages=[gender_indexer, phone_service_indexer, internet_service_indexer, churn_indexer,
                                 gender_encoder, phone_service_encoder, internet_service_encoder, assembler])
    
    preprocessed_df = pipeline.fit(df).transform(df)
    preprocessed_df.select("features", "ChurnIndexed").show()
    
    return preprocessed_df
```

#### Sample Output:
```text
+--------------------+-----------+
|features            |ChurnIndexed|
+--------------------+-----------+
|[0.0,12.0,29.85,...|0.0        |
|[0.0,1.0,56.95,... |1.0        |
|[1.0,5.0,53.85,... |0.0        |
+--------------------+-----------+
```

#### Command to Execute Task 1:
```bash
python customer_churn_analysis.py
```

---

### Task 2: Building and Training a Logistic Regression Model

In this task, a Logistic Regression model is built and evaluated to predict customer churn.

#### Steps:
1. **Split Data**: The data is split into training (80%) and test (20%) sets.
2. **Train Logistic Regression Model**: The model is trained using the `LogisticRegression` class.
3. **Evaluate Model**: The model’s performance is evaluated using AUC (Area Under ROC Curve).

#### Code:
```python
def train_logistic_regression_model(df):
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)
    lr = LogisticRegression(featuresCol="features", labelCol="ChurnIndexed")
    lr_model = lr.fit(train_data)
    
    predictions = lr_model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(labelCol="ChurnIndexed", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    
    print(f"AUC (Area Under ROC Curve): {auc}")
    predictions.select("Churn", "prediction", "probability").show(5)
```

#### Sample Output:
```text
AUC (Area Under ROC Curve): 0.86
+-----+----------+--------------------+
|Churn|prediction|           probability|
+-----+----------+--------------------+
|  Yes|       1.0|[0.2,0.8]|
|  No |       0.0|[0.7,0.3]|
+-----+----------+--------------------+
```

---

### Task 3: Feature Selection Using Chi-Square Test

In this task, feature selection is performed using the Chi-Square test to identify the most relevant features for predicting churn.

#### Steps:
1. **Apply Chi-Square Test**: The `ChiSqSelector` selects the top features based on the Chi-Square test.
2. **Display Selected Features**: The top 5 features are selected and displayed.

#### Code:
```python
def feature_selection(df):
    chi_selector = ChiSqSelector(featuresCol="features", labelCol="ChurnIndexed", outputCol="selected_features", numTopFeatures=5)
    selected_df = chi_selector.fit(df).transform(df)
    selected_df.select("Churn", "selected_features").show(5)
```

#### Sample Output:
```text
+-----+--------------------+
|Churn|    selected_features|
+-----+--------------------+
|  Yes|[1.0,0.0,0.0,0.0,1...|
|  No |[0.0,0.0,1.0,0.0,2...|
+-----+--------------------+
```

---

### Task 4: Hyperparameter Tuning and Model Comparison

In this task, we perform hyperparameter tuning using cross-validation and compare multiple models: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.

#### Steps:
1. **Split Data**: The data is split into training (80%) and test (20%) sets.
2. **Define Models and Hyperparameters**: Multiple models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) are defined, along with hyperparameter grids.
3. **Cross-Validation**: Cross-validation is applied to each model to find the best hyperparameters.
4. **Model Evaluation**: Each model's performance is evaluated using AUC.

#### Code:
```python
def tune_and_compare_models(df):
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)
    
    lr = LogisticRegression(featuresCol="features", labelCol="ChurnIndexed")
    dt = DecisionTreeClassifier(featuresCol="features", labelCol="ChurnIndexed")
    rf = RandomForestClassifier(featuresCol="features", labelCol="ChurnIndexed")
    gbt = GBTClassifier(featuresCol="features", labelCol="ChurnIndexed")

    # Define hyperparameter grids
    param_grid_lr = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1]).addGrid(lr.elasticNetParam, [0.5, 0.7]).build()
    param_grid_dt = ParamGridBuilder().addGrid(dt.maxDepth, [5, 10]).addGrid(dt.maxBins, [32, 64]).build()
    param_grid_rf = ParamGridBuilder().addGrid(rf.numTrees, [10, 20]).addGrid(rf.maxDepth, [5, 10]).build()
    param_grid_gbt = ParamGridBuilder().addGrid(gbt.maxIter, [10, 20]).addGrid(gbt.maxDepth, [5, 10]).build()

    # Define evaluator
    evaluator = BinaryClassificationEvaluator(labelCol="ChurnIndexed", metricName="areaUnderROC")

    # Cross-validation for each model
    cv_lr = CrossValidator(estimator=lr, estimatorParamMaps=param_grid_lr, evaluator=evaluator, numFolds=3)
    cv_dt = CrossValidator(estimator=dt, estimatorParamMaps=param_grid_dt, evaluator=evaluator, numFolds=3)
    cv_rf = CrossValidator(estimator=rf, estimatorParamMaps=param_grid_rf, evaluator=evaluator, numFolds=3)
    cv_gbt = CrossValidator(estimator=gbt, estimatorParamMaps=param_grid_gbt, evaluator=evaluator, numFolds=3)

    # Fit models with cross-validation
    cv_lr_model = cv_lr.fit(train_data)
    cv_dt_model = cv_dt.fit(train_data)
    cv_rf_model = cv_rf.fit(train_data)
    cv_gbt_model = cv_gbt.fit(train_data)

    # Make predictions on the test data
    lr_predictions = cv_lr_model.bestModel.transform(test_data)
    dt_predictions = cv_dt_model.bestModel.transform(test_data)
    rf_predictions = cv_rf_model.bestModel.transform(test_data)
    gbt_predictions = cv_gbt_model.bestModel.transform(test_data)

    # Evaluate models
    auc_lr = evaluator.evaluate(lr_predictions)
    auc_dt = evaluator.evaluate(dt_predictions)
    auc_rf = evaluator.evaluate(rf_predictions)
    auc_g

bt = evaluator.evaluate(gbt_predictions)

    print(f"AUC - Logistic Regression: {auc_lr}")
    print(f"AUC - Decision Tree: {auc_dt}")
    print(f"AUC - Random Forest: {auc_rf}")
    print(f"AUC - Gradient Boosting: {auc_gbt}")
```

---

### Conclusion

The model demonstrates how to preprocess customer data, train a classification model, perform feature selection, and compare various machine learning algorithms for predicting customer churn. Further improvements can be made by experimenting with other algorithms and performing more advanced feature engineering.