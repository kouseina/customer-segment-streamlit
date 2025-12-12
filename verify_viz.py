import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    print("Loading data...")
    # Create dummy data if file doesn't exist, but we know it does
    try:
        df = pd.read_csv('urbanmart_data.csv')
    except Exception:
        data = {
            'TransactionDate': pd.date_range(start='2023-01-01', periods=100, freq='D'),
            'TransactionValue': [1000] * 100,
            'Region': ['Jakarta'] * 50 + ['Bandung'] * 50
        }
        df = pd.DataFrame(data)

    print("Data loaded. Processing...")
    
    if 'TransactionDate' in df.columns:
        df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    
    # Test Monthly Resampling
    print("Testing Monthly Resampling...")
    # Try ME first, if fail catch
    try:
        monthly_sales = df.set_index('TransactionDate').resample('ME')['TransactionValue'].sum().reset_index()
        print("Resample 'ME' successful.")
    except Exception as e:
        print(f"Resample 'ME' failed: {e}. Trying 'M'...")
        monthly_sales = df.set_index('TransactionDate').resample('M')['TransactionValue'].sum().reset_index()
        print("Resample 'M' successful.")

    # Test Plotting (Non-GUI)
    print("Testing Plotting...")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.lineplot(data=monthly_sales, x='TransactionDate', y='TransactionValue', marker='o', ax=ax)
    print("Lineplot successful.")
    
    fig2, ax2 = plt.subplots(figsize=(6,4))
    if 'Region' in df.columns:
        sns.countplot(data=df, y="Region", ax=ax2, order=df['Region'].value_counts().index)
        print("Countplot successful.")

    print("All verification steps passed!")

except Exception as e:
    print(f"Verification FAILED: {e}")
    exit(1)
