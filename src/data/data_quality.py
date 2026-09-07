import pandas as pd

# -----------------------------------------
# STEP 1: Loading PaySim dataset
# -----------------------------------------

df = pd.read_csv("data/raw/PS_20174392719_1491204439457_log.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# -----------------------------------------
# STEP 2: Balance Consistency Investigation
# -----------------------------------------

# Origin account balance difference
df["origin_balance_diff"] = ( 
    df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
)

# Destination account balance difference
df["destination_balance_diff"] = (
    df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
)

# Summary statistics
print("\nOrigin Balance Difference:")
print(df["origin_balance_diff"].describe())

print("\nDestination Balance Difference:")
print(df["destination_balance_diff"].describe())

# -----------------------------------------
# STEP 3: Balance Consistency by Transaction Type
# -----------------------------------------

balance_check = (
    df.groupby("type").agg(
        transactions=("type","size"),
        origin_mean_diff=("origin_balance_diff","mean"),
        destination_mean_diff=("destination_balance_diff","mean"),
    ).round(2)
)

print("\nBalance Consistency by Transaction Type:")
print(balance_check)

# -----------------------------------------
# STEP 4: Zero Balance Investigation
# -----------------------------------------

balance_columns = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

zero_balance_summary = (
    df[balance_columns].eq(0).mean().mul(100).round(2)
)

print("\nPercentage of Zero Values in Balance Columns:")
print(zero_balance_summary)

# -----------------------------------------
# STEP 5: Zero Balances by Transaction Type
# -----------------------------------------

zero_by_type = (
    df.groupby("type")[balance_columns]
      .apply(lambda x: (x == 0).mean() * 100)
      .round(2)
)

print("\nPercentage of Zero Balances by Transaction Type:")
print(zero_by_type)

fraud_by_type = (
    df.groupby("type").agg(
        transactions=("isFraud","size"),
        fraud_transactions=("isFraud","sum"),
        fraud_rate=("isFraud","mean"),
    )
)

fraud_by_type["fraud_rate"] = (
    fraud_by_type["fraud_rate"] * 100
).round(4)

print("\nFraud Distribution by Transaction Type:")
print(fraud_by_type)