# Predicting Term Deposit Subscription

## Project Overview

This project aims to predict whether clients will subscribe to a term deposit using machine learning techniques. By leveraging client data, the goal is to optimize marketing strategies and enhance the effectiveness of marketing campaigns.

## Objective

- **Predict Client Subscription**: Identify whether clients are likely to subscribe to a term deposit based on various features.
- **Optimize Marketing**: Improve the targeting of marketing efforts to reduce costs and increase campaign efficiency.

## Dataset

The dataset used for this project is `bank-full.csv`, which contains information about bank clients and their interactions with marketing campaigns. The features include:

- `age`: Age of the client
- `job`: Type of job
- `marital`: Marital status
- `education`: Level of education
- `default`: Whether the client has credit in default
- `balance`: Account balance
- `housing`: Whether the client has a housing loan
- `loan`: Whether the client has a personal loan
- `contact`: Type of communication used to contact the client
- `day`: Last contact day of the month
- `month`: Last contact month of the year
- `duration`: Duration of the last contact
- `campaign`: Number of contacts performed during this campaign
- `pdays`: Number of days since the client was last contacted
- `previous`: Number of contacts performed before this campaign
- `poutcome`: Outcome of the previous marketing campaign
- `y`: Whether the client subscribed to a term deposit (target variable)

## Solution

### Data Exploration

- **Initial Exploration**: Analyzed feature distributions and relationships.
- **Data Cleaning**: Handled missing values and outliers.

### Preprocessing

- **Feature Engineering**: Encoded categorical variables and scaled numerical features.
- **Train-Test Split**: Divided the dataset into training and testing sets.

### Modeling

- **Algorithm**: Implemented a logistic regression model using a pipeline.
- **Evaluation**: Assessed model performance with metrics such as accuracy, precision, recall, and confusion matrix.

### Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
bank = pd.read_csv('bank-full.csv', delimiter=';')

# Preprocessing
X = bank.drop(columns=['contact', 'day', 'y'])
y = bank['y']

# Label Encoding
le = LabelEncoder()
y = le.fit_transform(y)

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), ['age', 'balance', 'campaign', 'pdays', 'previous']),
        ('cat', OneHotEncoder(sparse=False, drop='first'), ['job', 'marital', 'education', 'default', 'housing', 'loan', 'month', 'poutcome'])
    ],
    remainder='passthrough'
)

# Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()
