import pandas as pd

df = pd.read_csv("data/sensor_weather_reduced_timeseries.csv")

print("✅ Data loaded!")
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nData types:\n{df.dtypes}")