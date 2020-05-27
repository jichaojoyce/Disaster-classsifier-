"""
Documentation for Process_data.py
This is a Disaster Response Pipeline Project, Udacity - Data Science Nanodegree
Sample Script Execution:
> python process_data.py disaster_messages.csv disaster_categories.csv disease.db
Inputs:
    1) CSV file containing messages (disaster_messages.csv)
    2) CSV file containing categories (disaster_categories.csv)
Output:
    3) SQLite destination database (disease.db)
"""


import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load Data function
    This function is to merge the messages and categories datasets using the common id.
    Parameters:
        messages_filepath -> path to messages csv file
        categories_filepath -> path to categories csv file
    Return:
        df -> merged data as Pandas DataFrame
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df 

def clean_data(df):
    """
    Clean Data function
    This function is to split the values in the categories column on the ; character so that each value becomes a separate column. 
    Then define a new column using the first row of categories dataframe to create column names for the categories data.
    Continue, this function coverts category values to 0 or 1. 
    Finally a cleaned dataframe with messages and category values without duplicates values is created.
    Parameters:
        df -> raw data Pandas DataFrame
    Return:
        df -> clean data Pandas DataFrame
    """    
    
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    categories['related']=categories['related'].replace(2,0)    
    df=df.drop(['categories'], axis=1)
    df = pd.concat([df,categories], axis=1)
    df.drop_duplicates(inplace=True)
    return df 
def save_data(df, database_filename):
    """
    Save Data function
    This function is to save the clean dataset into an sqlite database
    Parameters:
        df -> Clean data Pandas DataFrame
        database_filename -> database file (.db) destination path
    """    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('df', engine, index=False)


def main():
    """
    Main Data Processing function
    
    This function implement the ETL pipeline:
        1) Data extraction from .csv
        2) Data cleaning and pre-processing
        3) Data saving to SQLite database
    """
    
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()