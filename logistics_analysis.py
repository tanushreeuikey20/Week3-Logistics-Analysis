
import pandas as pd
import numpy as np

# Load logistics dataset
file_name = "DataCoSupplyChainDataset.csv"

df = pd.read_csv(file_name, encoding="latin1")

# Display dataset information
print("LOGISTICS DATA ANALYSIS")
print("=" * 40)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

# Select numerical columns
numeric_data = df.select_dtypes(include="number")

print("\nDescriptive Statistics:")
print(numeric_data.describe().round(2))

print("\nMedian Values:")
print(numeric_data.median().round(2))

print("\nAnalysis completed successfully!")

import pandas as pd
import numpy as np

# Load logistics dataset
file_name = "DataCoSupplyChainDataset.csv"

df = pd.read_csv(file_name, encoding="latin1")

# Display dataset information
print("LOGISTICS DATA ANALYSIS")
print("=" * 40)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

# Select numerical columns
numeric_data = df.select_dtypes(include="number")

print("\nDescriptive Statistics:")
print(numeric_data.describe().round(2))

print("\nMedian Values:")
print(numeric_data.median().round(2))

print("\nAnalysis completed successfully!")


import os
import matplotlib.pyplot as plt
import seaborn as sns

# Create folder for graphs
os.makedirs("graphs", exist_ok=True)
sns.set_theme(style="whitegrid")

# Graph 1: Delivery Status
plt.figure(figsize=(9, 5))
df["Delivery Status"].value_counts().plot(kind="bar")
plt.title("Distribution of Delivery Status")
plt.xlabel("Delivery Status")
plt.ylabel("Number of Orders")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("graphs/01_delivery_status.png", dpi=300)
plt.close()

# Graph 2: Actual Shipping Days
plt.figure(figsize=(8, 5))
sns.histplot(df["Days for shipping (real)"].dropna(), bins=15, kde=True)
plt.title("Distribution of Actual Shipping Days")
plt.xlabel("Actual Shipping Days")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("graphs/02_shipping_days.png", dpi=300)
plt.close()

# Graph 3: Actual vs Scheduled Shipping Days
plt.figure(figsize=(8, 5))
sns.boxplot(data=df[[
    "Days for shipping (real)",
    "Days for shipment (scheduled)"
]])
plt.title("Actual vs Scheduled Shipping Time")
plt.ylabel("Days")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig("graphs/03_shipping_comparison.png", dpi=300)
plt.close()

# Graph 4: Shipping Mode
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="Shipping Mode",
    y="Days for shipping (real)",
    errorbar=None
)
plt.title("Average Shipping Time by Shipping Mode")
plt.xlabel("Shipping Mode")
plt.ylabel("Average Shipping Days")
plt.tight_layout()
plt.savefig("graphs/04_shipping_mode.png", dpi=300)
plt.close()

# Graph 5: Correlation Heatmap
cols = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Sales",
    "Order Item Quantity",
    "Benefit per order"
]

correlation = df[cols].corr(numeric_only=True)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Logistics Variables")
plt.tight_layout()
plt.savefig("graphs/05_correlation_heatmap.png", dpi=300)
plt.close()

print("All 5 graphs created successfully!")
print("Open the graphs folder to view them.")
