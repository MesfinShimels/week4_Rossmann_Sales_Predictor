import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(
    filename="eda.log", 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def clean_all_data(train, test, store, sample):
    """
    Cleans all datasets: train, test, store, and sample by handling missing values,
    converting data types, and standardizing dates.
    """
    logging.info("Starting data cleaning process for all datasets.")

    # Helper function to fill missing values
    def fill_missing_values(df, strategy='mean'):
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:  # Numeric columns
                if strategy == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
            elif df[col].dtype == 'object':  # Categorical columns
                df[col].fillna(df[col].mode()[0], inplace=True)
        return df

    # Clean train data
    train['Date'] = pd.to_datetime(train['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    train['StateHoliday'] = train['StateHoliday'].astype(str)
    train = fill_missing_values(train, strategy='mean')
    logging.info("Cleaned train data.")

    # Clean test data
    test['Date'] = pd.to_datetime(test['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    test['StateHoliday'] = test['StateHoliday'].astype(str)
    test = fill_missing_values(test, strategy='mean')
    logging.info("Cleaned test data.")

    # Clean store data
    store['StoreType'] = store['StoreType'].astype(str)
    store['Assortment'] = store['Assortment'].astype(str)
    store['PromoInterval'] = store['PromoInterval'].fillna(store['PromoInterval'].mode()[0])
    store = fill_missing_values(store, strategy='mean')
    logging.info("Cleaned store data.")

    # Ensure sample data
    if not {'Id', 'Sales'}.issubset(sample.columns):
        logging.warning("Sample data is missing required columns ('Id', 'Sales'). Please verify the file structure.")
    sample.fillna(0, inplace=True)
    logging.info("Ensured sample data has no missing values by filling with 0.")

    return train, test, store, sample


def check_promo_distribution(train, test):
    """
    Compares the distribution of promos in train and test datasets.
    """
    logging.info("Checking promo distribution in training and test datasets.")
    
    train_promo_dist = train['Promo'].value_counts(normalize=True)
    test_promo_dist = test['Promo'].value_counts(normalize=True)
    
    logging.info("Promo distribution in training dataset:\n" + str(train_promo_dist))
    logging.info("Promo distribution in test dataset:\n" + str(test_promo_dist))
    
    promo_dist_df = pd.DataFrame({
        "Train": train_promo_dist,
        "Test": test_promo_dist
    }).fillna(0)
    
    promo_dist_df.plot(kind='bar', figsize=(10, 6), color=['blue', 'orange'])
    plt.title('Promo Distribution: Train vs Test')
    plt.xlabel('Promo')
    plt.ylabel('Proportion')
    plt.xticks(rotation=0)
    plt.show()

def sales_during_holidays(train):
    """
    Compares sales behavior before, during, and after holidays.
    """
    logging.info("Analyzing sales behavior during holidays.")
    
    train['Date'] = pd.to_datetime(train['Date'])
    train['IsHoliday'] = train['StateHoliday'] != '0'
    
    sales_summary = train.groupby('IsHoliday')['Sales'].mean()
    logging.info("Sales during holidays:\n" + str(sales_summary))
    
    sales_summary.plot(kind='bar', figsize=(8, 5), color='green')
    plt.title('Sales During Holidays')
    plt.xlabel('Holiday (True/False)')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=0)
    plt.show()

def seasonal_sales_analysis(train):
    """
    Analyzes seasonal (Christmas, Easter) purchase behaviors.
    """
    logging.info("Analyzing seasonal sales trends.")
    
    train['Date'] = pd.to_datetime(train['Date'])
    train['Month'] = train['Date'].dt.month
    monthly_sales = train.groupby('Month')['Sales'].mean()
    
    monthly_sales.plot(kind='line', figsize=(10, 6), marker='o', color='red')
    plt.title('Seasonal Sales Trends')
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    plt.grid()
    plt.show()

def analyze_correlation(train):
    """
    Analyzes correlation between sales and the number of customers.
    """
    logging.info("Analyzing correlation between sales and customers.")
    
    correlation = train[['Sales', 'Customers']].corr()
    logging.info("Correlation matrix:\n" + str(correlation))
    
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()
    return correlation

def analyze_promo_effectiveness(train):
    """
    Analyzes how promotions affect sales and customer behavior.
    """
    logging.info("Analyzing the effect of promotions on sales.")
    
    promo_effect = train.groupby('Promo')['Sales'].mean()
    promo_customers = train.groupby('Promo')['Customers'].mean()
    
    logging.info("Promo effect on sales:\n" + str(promo_effect))
    logging.info("Promo effect on customers:\n" + str(promo_customers))
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    promo_effect.plot(kind='bar', ax=ax[0], color='skyblue')
    ax[0].set_title('Promo Effect on Sales')
    ax[0].set_xlabel('Promo')
    ax[0].set_ylabel('Average Sales')
    
    promo_customers.plot(kind='bar', ax=ax[1], color='orange')
    ax[1].set_title('Promo Effect on Customers')
    ax[1].set_xlabel('Promo')
    ax[1].set_ylabel('Average Customers')
    
    plt.tight_layout()
    plt.show()

def assortment_analysis(store, train):
    """
    Analyzes how assortment types affect sales.
    """
    logging.info("Analyzing the effect of assortment types on sales.")
    
    store_sales = pd.merge(train, store, on='Store')
    assortment_sales = store_sales.groupby('Assortment')['Sales'].mean()
    logging.info("Assortment effect on sales:\n" + str(assortment_sales))
    
    assortment_sales.plot(kind='bar', figsize=(10, 6), color='purple')
    plt.title('Assortment Effect on Sales')
    plt.xlabel('Assortment Type')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=0)
    plt.show()

def competitor_distance_analysis(store, train):
    """
    Analyzes the effect of competitor distance on sales.
    """
    logging.info("Analyzing competitor distance effect on sales.")
    
    store_sales = pd.merge(train, store, on='Store')
    store_sales['CompetitionDistance'].fillna(store_sales['CompetitionDistance'].max(), inplace=True)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=store_sales, x='CompetitionDistance', y='Sales', alpha=0.5)
    plt.title('Competition Distance vs Sales')
    plt.xlabel('Competition Distance (meters)')
    plt.ylabel('Sales')
    plt.show()
