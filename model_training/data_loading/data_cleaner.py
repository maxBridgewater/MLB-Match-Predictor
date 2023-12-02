import pandas as pd


def create_mapping_dict(filepath):

    map = {}

    with open(filepath, 'r') as file:
        for entry in file:
            key = entry.strip()
            val = len(map)
            map[key] = val

    return map


def get_cleaned_data(filepath_data, filepath_teams):

    #Load the match data
    df = pd.read_csv(filepath_data, header=0)

    #Create the dict to map team names to integers
    map = create_mapping_dict(filepath_teams)
    
    df.replace(map, inplace=True)

    df['date'] = pd.to_datetime(df['date'])

    #Convert the date time column into days elapsed since Jan 1st 2022
    start_date = pd.to_datetime('2022-01-01')

    df['days_elapsed'] = (df['date'] - start_date).dt.days

    df.drop('date', axis=1, inplace=True)

    return df

