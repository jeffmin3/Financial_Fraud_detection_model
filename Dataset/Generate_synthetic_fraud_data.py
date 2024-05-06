from faker import Faker
import random
import pandas as pd
from datetime import datetime, timedelta

# Initialize Faker generator
fake = Faker()

# Generate synthetic data for feature 1 (transaction amount)
feature1 = [round(random.uniform(1, 1000), 2) for _ in range(4000)]

# Generate synthetic data for feature 2 (transaction timestamp)
start_date = datetime(2020, 1, 1)
end_date = datetime(2022, 12, 31)
timestamps = [(start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))).strftime('%Y-%m-%d %H:%M:%S') for _ in range(4000)]

# Generate synthetic data for other attributes
transaction_ids = list(range(1, 4001))
merchant_ids = [fake.uuid4()[:8] for _ in range(4000)]  # Generating random merchant IDs
customer_ids = [fake.uuid4()[:8] for _ in range(4000)]  # Generating random customer IDs
is_fraud = [random.choice([0, 1]) for _ in range(4000)]  # Randomly assigning 0 or 1 for is_fraud

# Create DataFrame
data = pd.DataFrame({
    'transaction_id': transaction_ids,
    'timestamp': timestamps,
    'amount': feature1,
    'merchant_id': merchant_ids,
    'customer_id': customer_ids,
    'is_fraud': is_fraud
})

# Display the first few rows of the generated dataset
print(data.head())

# Save the dataset to a CSV file
data.to_csv('synthetic_fraud_dataset.csv', index=False)
