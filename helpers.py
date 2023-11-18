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
    