import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def read_and_describe(path):
    df = pd.read_csv(path, encoding='ISO-8859-1', parse_dates=['InvoiceDate']) # Windows-125 works as well

    print('Data types:\n')
    print(df.info())

    print('\nNumeric data description:\n')
    print(df.describe())

    print('\nMissing data:\n')
    print(df.isna().sum())
    return df;

def description_stock_mapping(df):
    print(f'Number of unique descriptions {df["Description"].nunique()}')
    print(f'Number of unique StockCode {df["StockCode"].nunique()}')

    # Number of descriptions with the same stockCode
    unique_counts = df.groupby("Description")["StockCode"].nunique().sort_values(ascending=False)
    display(unique_counts)

    # Count the number of occurrences for each unique count value
    return unique_counts.value_counts().sort_index(ascending=False)

def calculate_frequent_patterns(df, min_support):
    # Each transaction is a list of items bought together
    transactions = df.groupby('InvoiceNo')['Description'].apply(list).tolist()

    # Initialize TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)

    # Transform into DataFrame for easier use with mlxtend
    df_transactions = pd.DataFrame(te_ary, columns=te.columns_)

    # Apply the Apriori algorithm with a specified support value, e.g., 0.01 (1%)
    frequent_itemsets = apriori(df_transactions, min_support, use_colnames=True)
    return frequent_itemsets

def generate_association_rules(fp, min_confidence, min_lift = 1):
    # Generate association rules
    rules = association_rules(fp, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules['lift'] >= min_lift]
    rules['rule'] = (
        rules['antecedents'].apply(lambda x: set(x)).astype(str) +
        " => " +
        rules['consequents'].apply(lambda x: set(x)).astype(str)
    )
    selected_columns = ['rule', 'support', 'confidence', 'lift']

    # Create a new DataFrame containing only the selected columns
    return rules[selected_columns]

def generate_association_rules_hybrid(fp, min_confidence, confidence_weight):
    # Generate association rules
    rules = association_rules(fp, metric="confidence", min_threshold=min_confidence)

    # Normalize lift using Min-Max Normalization
    min_lift = rules['lift'].min()
    max_lift = rules['lift'].max()
    rules['normalized_lift'] = (rules['lift'] - min_lift) / (max_lift - min_lift)

    # Combine confidence and normalized lift with equal weights
    rules['combined_score'] = confidence_weight * rules['confidence'] + (1-confidence_weight) * rules['normalized_lift']

    # Create the rule as a string
    rules['rule'] = (
        rules['antecedents'].apply(lambda x: set(x)).astype(str) +
        " => " +
        rules['consequents'].apply(lambda x: set(x)).astype(str)
    )

    # Select the desired columns
    selected_columns = ['rule', 'support', 'confidence', 'lift', 'combined_score']

    # Create a new DataFrame containing only the selected columns
    return rules[selected_columns]