import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime



df = pd.read_csv('dataset/sales.csv')
print(df.head())

print("Dataset Shape:")
print(df.shape)

print("Columns:")
print(df.columns)

print(df.dtypes)

print(df.info())

print(df.describe())

print("Missing Values:")
print(df.isnull().sum())

print("Duplicate Rows:")
print(df.duplicated().sum())
print(df.nunique())

plt.figure(figsize=(8,5))
sns.boxplot(x=df['Sales'])
plt.title('Sales Outliers')
plt.show()


print(df.duplicated().sum())
df = df.drop_duplicates()
print(df.duplicated().sum())

df = df.dropna()
df['Sales'] = df['Sales'].fillna(df['Sales'].mean())
df['Category'] = df['Category'].fillna(df['Category'].mode()[0])

df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(' ', '_')

print(df.columns)

df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True)

print(df['order_date'].head())

df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month

def sales_category(x):
    if x > 500:
        return 'High'
    elif x > 200:
        return 'Medium'
    else:
        return 'Low'


df['sales_category'] = df['sales'].apply(sales_category)

Q1 = df['sales'].quantile(0.25)
Q3 = df['sales'].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR


df = df[(df['sales'] >= lower_limit) & (df['sales'] <= upper_limit)]

plt.figure(figsize=(8,5))
plt.hist(df['sales'], bins=20)
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

category_sales = df.groupby('category')['sales'].sum()

category_sales.plot(kind='bar')
plt.title('Category Wise Sales')
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.show()

df.to_csv('output/cleaned_sales.csv', index=False)

print("Cleaned dataset saved successfully.")

print("Final Dataset Shape:", df.shape)
print("Missing Values After Cleaning:")
print(df.isnull().sum())