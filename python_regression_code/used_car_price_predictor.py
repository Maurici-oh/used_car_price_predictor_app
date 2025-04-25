import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --------------------------------
# Configuration and Data Loading
# --------------------------------

# Configure Seaborn aesthetics
sns.set_context()

# Load dataset
raw_data = pd.read_csv('Used_cars_data.csv')

# Show general dataset info
print(raw_data.describe(include='all'))
print(raw_data.info())

# --------------------------------
# Data Cleaning
# --------------------------------

# Drop 'Model' column due to high cardinality
data = raw_data.drop(['Model'], axis=1)
print(data.describe(include='all'))

# Check and drop rows with missing values
print(data.isnull().sum())
data_no_mv = data.dropna(axis=0)
print(data_no_mv.describe(include='all'))

# ------------------------------
# Exploratory Data Analysis (EDA)
# ------------------------------

# Plot distribution of Price
sns.histplot(data_no_mv['Price'], kde=True)
plt.title('Price Distribution')
plt.show()

# Remove top 1% Price outliers
q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price'] < q]
sns.histplot(data_1['Price'], kde=True)
plt.title('Price Distribution (After Removing Top 1% Outliers)')
plt.show()

# Remove top 1% Mileage outliers
sns.histplot(data_no_mv['Mileage'], kde=True)
q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage'] < q]
sns.histplot(data_2['Mileage'], kde=True)
plt.title('Mileage Distribution (After Removing Top 1% Outliers)')
plt.show()

# Remove anomalous EngineV values
sns.histplot(data_no_mv['EngineV'], kde=True)
data_3 = data_2[data_2['EngineV'] < 6.5]
sns.histplot(data_3['EngineV'], kde=True)
plt.title('Engine Volume Distribution (After Cleaning)')
plt.show()

# Remove bottom 1% of Year values
sns.histplot(data_no_mv['Year'], kde=True)
q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year'] > q]
sns.histplot(data_4['Year'], kde=True)
plt.title('Year Distribution (After Cleaning)')
plt.show()

# Reset index after cleaning
data_cleaned = data_4.reset_index(drop=True)

# -------------------------------------
# Visual Inspection for Linearity
# -------------------------------------

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 3))
ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
ax1.set_title('Price vs Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
ax2.set_title('Price vs EngineV')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
ax3.set_title('Price vs Mileage')
plt.show()

# Log-transform target to improve linearity
data_cleaned['log_price'] = np.log(data_cleaned['Price'])

# Re-visualize with log-transformed price
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 3))
ax1.scatter(data_cleaned['Year'], data_cleaned['log_price'])
ax1.set_title('Log(Price) vs Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['log_price'])
ax2.set_title('Log(Price) vs EngineV')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['log_price'])
ax3.set_title('Log(Price) vs Mileage')
plt.show()

# Drop original Price column
data_cleaned = data_cleaned.drop(['Price'], axis=1)

# ---------------------------
# Multicollinearity Check
# ---------------------------

variables = data_cleaned[['Mileage', 'Year', 'EngineV']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
print(vif)

# Drop 'Year' to reduce multicollinearity
data_no_multicollinearity = data_cleaned.drop(['Year'], axis=1)

# --------------------------
# Categorical Encoding
# --------------------------

data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)

# Define column order
cols = [
    'log_price', 'Mileage', 'EngineV', 'Brand_BMW', 'Brand_Mercedes-Benz',
    'Brand_Mitsubishi', 'Brand_Renault', 'Brand_Toyota', 'Brand_Volkswagen',
    'Body_hatch', 'Body_other', 'Body_sedan', 'Body_vagon', 'Body_van',
    'Engine Type_Gas', 'Engine Type_Other', 'Engine Type_Petrol',
    'Registration_yes'
]
data_preprocessed = data_with_dummies[cols]

# --------------------------
# Linear Regression Modeling
# --------------------------

# Define target and input features
targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'], axis=1)

# Standardize inputs
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=42)

# Train linear regression model
reg = LinearRegression()
reg.fit(x_train, y_train)

# Predictions on training set
y_hat = reg.predict(x_train)

# Predicted vs Actual (Training)
plt.scatter(y_train, y_hat)
plt.xlabel('Actual Log Prices')
plt.ylabel('Predicted Log Prices')
plt.title('Predicted vs Actual (Training Set)')
plt.xlim(6, 13)
plt.ylim(6, 13)
plt.show()

# Plot residuals
sns.histplot(y_train - y_hat, kde=True)
plt.title('Residuals Distribution')
plt.show()

# Model performance
print(f"R² Score (Train): {reg.score(x_train, y_train):.3f}")

# Model coefficients
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
print(reg_summary)

# ---------------------
# Model Testing Phase
# ---------------------

# Predictions on test set
y_hat_test = reg.predict(x_test)

# Compare predictions to actual values
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Actual Log Prices')
plt.ylabel('Predicted Log Prices')
plt.title('Predicted vs Actual (Test Set)')
plt.xlim(6, 13)
plt.ylim(6, 13)
plt.show()

# Build performance dataframe
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
df_pf['Target'] = np.exp(y_test.reset_index(drop=True))
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
df_pf['Difference%'] = np.abs(df_pf['Residual'] / df_pf['Target'] * 100)

# Display performance results
pd.options.display.max_rows = 100
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(df_pf.sort_values(by=['Difference%']))
print(df_pf.describe())
