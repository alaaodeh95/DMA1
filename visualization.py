import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def data_visualization(df, percentile=1, bins=50):
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Filtering based on percentile
    quantity_threshold = df['Quantity'].quantile(percentile)
    unitprice_threshold = df['UnitPrice'].quantile(percentile)
    filtered_df = df[(df['Quantity'] <= quantity_threshold) & (df['UnitPrice'] <= unitprice_threshold)]

    # Creating a grid of subplots
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

    # Histogram for Quantity
    filtered_df['Quantity'].hist(bins=bins, ax=axes[0,0])
    axes[0,0].set_title('Distribution of Quantity')
    axes[0,0].set_xlabel('Quantity')
    axes[0,0].set_ylabel('Frequency')

    # Histogram for UnitPrice
    filtered_df['UnitPrice'].hist(bins=bins, ax=axes[0,1])
    axes[0,1].set_title('Distribution of UnitPrice')
    axes[0,1].set_xlabel('UnitPrice')
    axes[0,1].set_ylabel('Frequency')

    # Bar chart for Countries
    df['Country'].value_counts().plot(kind='bar', ax=axes[0,2])
    axes[0,2].set_title('Number of Transactions per Country')
    axes[0,2].set_xlabel('Country')
    axes[0,2].set_ylabel('Number of Transactions')

    # Timeline plot for InvoiceDate
    df.set_index('InvoiceDate').resample('M').size().plot(ax=axes[1,0])
    axes[1,0].set_title('Number of Transactions Over Time')
    axes[1,0].set_xlabel('Date')
    axes[1,0].set_ylabel('Number of Transactions')

    # Boxplot for Quantity
    sns.boxplot(x=filtered_df['Quantity'], ax=axes[1,1])
    axes[1,1].set_title('Boxplot of Quantity')

    # Boxplot for UnitPrice
    sns.boxplot(x=filtered_df['UnitPrice'], ax=axes[1,2])
    axes[1,2].set_title('Boxplot of UnitPrice')


    # Density Plot for UnitPrice
    sns.kdeplot(filtered_df['UnitPrice'], fill=True, ax=axes[2,0])
    axes[2,0].set_title('Density Plot of UnitPrice')
    axes[2,0].set_xlabel('UnitPrice')

    # Scatter Plot for Quantity vs UnitPrice
    axes[2,1].scatter(filtered_df['Quantity'], filtered_df['UnitPrice'])
    axes[2,1].set_title('Scatter Plot of Quantity vs UnitPrice')
    axes[2,1].set_xlabel('Quantity')
    axes[2,1].set_ylabel('UnitPrice')
    
    # Bar chart for Items
    df['Description'].value_counts().head(30).plot(kind='bar', ax=axes[2,2])
    axes[2,2].set_title('Most frequent items')
    axes[2,2].set_xlabel('Description')
    axes[2,2].set_ylabel('Number of Items in transactions')

    plt.tight_layout() # Adjusts the plots to fit in the figure area
    plt.show()
    
def sales_for_top_countries(df):
    # Determine top 10 countries by total sales
    top_countries = df.groupby('Country')['TotalPrice'].sum().nlargest(10).index

    # Filter the dataset for top 10 countries
    top_countries_df = df[df['Country'].isin(top_countries)]

    # Group by Country and Month, then sum up the TotalPrice
    monthly_sales_by_country = top_countries_df.groupby([top_countries_df['Country'], top_countries_df['InvoiceDate'].dt.to_period('M')])['TotalPrice'].sum()

    # Calculate the cumulative sum for each country
    cumulative_monthly_sales_by_country = monthly_sales_by_country.groupby(level=0).cumsum()

    # Plot
    plt.figure(figsize=(15/2, 8/2))
    for country in cumulative_monthly_sales_by_country.index.get_level_values(0).unique():
        cumulative_monthly_sales_by_country[country].plot(kind='line', marker='o', label=country)

    plt.title('Cumulative Total Sales Per Month by Country for Top 10 Countries (Log Scale)')
    plt.xlabel('Month')
    plt.ylabel('Cumulative Sales')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def customer_trend_per_month(df):
    # Determine the first purchase date for each customer
    first_purchase = df.groupby('CustomerID')['InvoiceDate'].min().reset_index()
    first_purchase.columns = ['CustomerID', 'FirstPurchaseDate']
    
    df = pd.merge(df, first_purchase, on='CustomerID')

    # Classify each transaction
    df['CustomerType'] = 'Retained'
    df.loc[df['InvoiceDate'] == df['FirstPurchaseDate'], 'CustomerType'] = 'New'

    # Aggregate data by month for new and retained customers
    monthly_customers = df.groupby([pd.Grouper(key='InvoiceDate', freq='M'), 'CustomerType']).nunique()['CustomerID']

    # Aggregate data by month for total customers
    total_monthly_customers = df.groupby(pd.Grouper(key='InvoiceDate', freq='M')).nunique()['CustomerID']

    # Unstack for plotting new and retained customers
    monthly_customers = monthly_customers.unstack(level=1).fillna(0)

    # Plot
    plt.figure(figsize=(15/2, 8/2))

    # Plot for new customers
    monthly_customers['New'].plot(kind='line', marker='o', label='New Customers', color='green')

    # Plot for retained customers
    monthly_customers['Retained'].plot(kind='line', marker='o', label='Retained Customers', color='blue')

    # Plot for total customers
    total_monthly_customers.plot(kind='line', marker='o', label='Total Customers', color='purple')

    plt.title('Customer Trends Per Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Customers')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def top_selling_items(df):
    # Calculate total sales for each item in each country
    item_sales_by_country = df.groupby(['Country', 'Description'])['TotalPrice'].sum().reset_index()

    # Determine the top-selling item for each country
    top_selling_items = item_sales_by_country.sort_values('TotalPrice', ascending=False).drop_duplicates(['Country'])

    # Sort countries for better visualization
    top_selling_items = top_selling_items.sort_values('Country')

    # Create a bar chart
    plt.figure(figsize=(10, 7))
    plt.bar(top_selling_items['Country'], top_selling_items['TotalPrice'])
    plt.xlabel('Country')
    plt.ylabel('Total Sales of Top Item')
    plt.title('Top Selling Item in Each Country')

    # Add the top item's description as labels on the bars
    for index, value in enumerate(top_selling_items['TotalPrice']):
        plt.text(index, value, str(top_selling_items.iloc[index]['Description']), rotation=90, va='bottom', ha='center')

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
