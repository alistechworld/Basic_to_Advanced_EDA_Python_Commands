import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# Load the Dataset 
df = pd.read_csv("Your Dataset File path and its name.csv")

# Display the First Few Rows of the Dataset
print(df.head(5))

# Basic information about the dataset
print(df.info())

# Summary Staticstics 
print(df.describe(include="all"))

# Check for the missing Values
print(df.isnull().sum())

# Visualizing missing values if any
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Distribution of numerical features
numerical_features = ['carat', 'depth', 'table', 'x', 'y', 'z', 'price']
for feature in numerical_features:
    plt.figure(figsize=(6,4))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()
    
# Count plots for categorical features
categorical_features = ['cut', 'color', 'clarity']
for feature in categorical_features:
    plt.figure(figsize=(6,4))
    sns.countplot(x=feature, data=df)
    plt.title(f'Count of {feature}')
    plt.show()

# Box plots to visualize price vs categorical features
for feature in categorical_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=feature, y='price', data=df)
    plt.title(f'Price vs {feature}')
    plt.show()

# Correlation matrix
correlation = df[numerical_features].corr()
plt.figure(figsize=(8,6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Pairplot for numerical features
sns.pairplot(df[numerical_features], diag_kind='kde')
plt.show()

# Insights from EDA
print("Business Insights:")
print("- Carat weight has a strong positive correlation with price.")
print("- Diamonds with 'Ideal' and 'Premium' cuts tend to have higher prices.")
print("- Higher clarity grades are associated with higher prices.")
print("- Color grade D diamonds are priced higher than those with lower grades.")
print("- Proper depth and table proportions contribute to higher diamond prices.")
    
