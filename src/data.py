import pandas as pd
from src import utils
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from numpy import mean

"""
    output: raw data using the csv file
    --
    We load the data from covid.csv by default. It is the data downloaded from
    https://github.com/beoutbreakprepared/nCoV2019/tree/master/latest_data
    The file covid.csv (available in /resources/covid.csv) has been downloaded on 20.04.2020.
"""


def loadData():
    return read_csv('./resources/covid.csv', header=0,
                    dtype={'ID': str, 'age': str, 'sex': str, 'city': str, 'province': str, 'country': str,
                           'latitude': float, 'longitude': float, 'geo_resolution': str, 'date_onset_symptoms': str,
                           'date_admission_hospital': str, 'date_confirmation': str, 'symptoms': str,
                           'lives_in_Wuhan': str,
                           'travel_history_dates': str, 'travel_history_location': str, 'reported_market_exposure': str,
                           'additional_information': str, 'chronic_disease_binary': bool, 'chronic_disease': str,
                           'source': str, 'sequence_available': str, 'outcome': str, 'date_death_or_discharge': str,
                           'notes_for_discussion': str, 'location': str, 'admin3': str, 'admin2': str, 'admin1': str,
                           'country_new': str, 'admin_id': float, 'data_moderator_initials': str,
                           'travel_history_binary': str})


"""
    input: string data in column 'travel_history_location'
    output: integer data according to whether or not the person has visited Wuhan
    --
    Returns 1 if the person has visited Wuhan according to the database, returns 0 otherwise.
    If the data is missing, we will consider that the person did not visit Wuhan.
"""


def traveledToWuhan(travel_history):
    if 'wuhan' in travel_history.lower():
        return 1
    else:
        return 0


"""
    input: string data in column 'date_onset_symptoms'
    output: integer data according to whether or not the person has visited Wuhan
    --
    Returns 1 if the person has visited Wuhan according to the database, returns 0 otherwise.
    If the data is missing, we will consider that the person did not visit Wuhan.
"""


def gotCoroned(dateOfCoronavirus):
    if dateOfCoronavirus != 'nan':
        return 1
    else:
        return 0

"""
    input: age or age range
    output: age or mean of the age range
    --
    For computation purpose, we need to get fix values. In case we get a range we will compute a mean. If we get an
    estimation, we will take the estimation boundaries.
"""


def to_float(list_str):
    if len(list_str) > 1 and (list_str[1] == "" or list_str[1] is None):
        return float(list_str[0])
    else:
        return mean([float(x) for x in list_str])


"""
    input: raw uncleaned data
    output: data used for part 1: Analysis of the dataset
    --
    For the part on the analysis of the dataset, we only need some data:
    { 'age', 'sex', 'outcome', 'country', 'chronic_disease_binary' }
    We trim the data to enable PCA and plotting of the data, as raw data was difficult to use.
"""


def trim(raw, norm=True):
    """
        Cleaning dataframe: we only use useful columns. List of data considered useful:
        { age, sex, country, chronic_disease_binary, outcome }
        For cleaning columns: https://realpython.com/python-data-cleaning-numpy-pandas/
    """
    data = raw.copy(deep=True)
    new_data = data.get(['age', 'sex', 'outcome', 'country', 'chronic_disease_binary'])  # get useful columns
    new_data = new_data.dropna()  # drops columns with incomplete data
    new_data = new_data.reset_index()  # reset indexes for later uses
    new_data = new_data.drop(['index'], axis=1)


    """
        Targeting: we add the target column (outcome).
        Cleaning the column:
            Any data in the death dictionary should be considered as (1).
            People that survived should be considered as (0).
            For later iterations, we could work with 3 states (dead, in hospital, recovered)
    """
    #  Death dictionary is defined in utils.py as __death__
    death_index = new_data['outcome'].isin(utils.__death__)
    #  Adding the outcome column to the dataframe
    new_data['outcome'][death_index] = 1
    new_data['outcome'][~death_index] = 0

    """
        Normalization of data
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    """
    if norm:
        new_data['country'] = new_data['country'].astype('category').cat.codes
        new_data['age'] = [to_float(x) for x in new_data['age'].astype(str).str.split('-')]
        new_data['sex'] = new_data['sex'].astype('category').cat.codes
        scaled = StandardScaler().fit_transform(new_data.get(['age', 'sex', 'outcome', 'country', 'chronic_disease_binary']))
        new_data = pd.DataFrame(scaled, index=new_data.index, columns=new_data.columns)

    return new_data


"""
    input: raw uncleaned data
    output: data used for part 2: Bayes Nets
    --
    For the part on Bayes Nets, we only need some data:
    { 'date_confirmation', 'date_onset_symptoms', 'travel_history_location', 'outcome', date_death_or_discharge }
"""


def BN_data(raw):
    data = raw.copy(deep=True)
    new_data = data.get(
        ['date_confirmation', 'date_onset_symptoms', 'travel_history_location', 'outcome', 'date_death_or_discharge'])
    # As we build the columns using lack of data as a data (false), no need to drop incomplete columns using dropna()
    new_data['travel_history_location'] = [traveledToWuhan(x) for x in
                                           new_data['travel_history_location'].astype(str)]
    new_data['date_onset_symptoms'] = [gotCoroned(x) for x in new_data['date_onset_symptoms'].astype(str)]

    #  Death dictionary is defined in utils.py as __death__
    death_index = new_data['outcome'].isin(utils.__death__)
    #  Adding the outcome column to the dataframe
    new_data['outcome'][death_index] = 1
    new_data['outcome'][~death_index] = 0

    new_data = new_data.dropna()  # Making sure no row is missing data, though it should not do anything
    return new_data
