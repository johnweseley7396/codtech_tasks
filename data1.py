# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset (available directly in seaborn)
df = sns.load_dataset('iris')

# 1. Display basic info about the dataset
print("Dataset Info:")
df.info()

# 2. Display first few rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(df.head())

# 3. Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# 4. Check for Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 5. Visualizing the Distribution of Numerical Features (Histograms)
plt.figure(figsize=(12, 8))
df.drop(columns='species').hist(bins=15, color='skyblue', edgecolor='black')
plt.suptitle('Histograms of Numerical Features')
plt.show()

# 6. Correlation Analysis: Correlation Matrix and Heatmap
correlation_matrix = df.drop(columns='species').corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plotting the heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, linecolor='black')
plt.title('Correlation Matrix Heatmap')
plt.show()

# 7. Outlier Detection with Box Plots
plt.figure(figsize=(12, 8))
sns.boxplot(data=df.drop(columns='species'))
plt.title('Box Plots for Outlier Detection')
plt.xticks(rotation=45)
plt.show()

# 8. Visualize Relationships Between Features with Pairplot
sns.pairplot(df, hue='species')
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()

# 9. Scatter Plot to visualize relationships (Sepal Length vs Petal Length)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=df, palette='Set1')
plt.title('Scatter Plot: Sepal Length vs Petal Length')
plt.show()

# 10. Distribution of Petal Length with KDE Plot
plt.figure(figsize=(8, 6))
sns.kdeplot(df['petal_length'], shade=True, color='orange')
plt.title('Distribution of Petal Length (KDE)')
plt.show()
