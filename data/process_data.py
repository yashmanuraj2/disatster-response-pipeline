

''' RUN THE FILE USING THE FOLLOWING COMMAND IN terminal cd/data:
    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db'''


import sys
import pandas as pd
import pandas as pd
from sqlalchemy import create_engine
''' Loads the messages and categories dataset and creates a new df by combining them on id '''
def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
   

    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(left=messages ,right=categories, left_on='id', right_on='id')
    return df
    ''' forms a new column for every 36 categories in the dataset and stores it into categries dataset '''
def clean_data(df):
    
    
    pd.options.display.max_columns = 4000
    categories= df["categories"].str.split(";",n=36,expand = True) 
    categories.head()
# select the first row of the categories dataframe
    row = categories.iloc[0,:]
# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x:x[:-2])
    
    
    
    
# rename the columns of `categories`

    categories.columns = category_colnames

    

    for column in categories:
    # set each value to be the last character of the string
         categories[column] = categories[column].str[-1]
    
    # convert column from string to numeric
         categories[column] = categories[column].astype(int)
      
    df = df.drop('categories',axis=1)
         
    df = pd.concat([df,categories],axis=1)
         
    print('Number of duplicated rows: {} out of {} samples'.format(df.duplicated().sum(),df.shape[0]))
    df = df.drop_duplicates()
    print('Number of duplicated rows: {} out of {} samples'.format(df.duplicated().sum(),df.shape[0]))
    return df
    
def save_data(df, database_filename):
    
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False,if_exists = 'replace')
 
'''Runs all the storing cleaning steps and saves the dataset into a new Table DisasterResponse'''
def main():
    
    if len(sys.argv) == 4:
     
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print('Cleaning data...')
        df = clean_data(df)
        print(df)
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
