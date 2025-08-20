
import pandas as pd
import numpy as np

# Sample data
data = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Math": [85, 70, 95, 60],
    "Science": [88, 75, 92, 65]
}

# Create DataFrame
df = pd.DataFrame(data)

print("Student Marks:")
print(df)

# Calculate average marks
df["Average"] = np.mean(df[["Math", "Science"]], axis=1)

print("\nWith Average:")
print(df)

# Find the topper
topper = df.loc[df["Average"].idxmax()]
print(f"\nTopper is {topper['Name']} with average {topper['Average']}")