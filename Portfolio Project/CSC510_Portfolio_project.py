#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import keras
from sklearn.preprocessing import MinMaxScaler


def scale_data(dataframe):
    '''Accepts data frames and applies min-max scaler
    return: scaled pandas dataframe'''
    
    scaler = MinMaxScaler()
    scaler.fit(dataframe)
    
    df_scaled = scaler.transform(dataframe)
    df_scaled = pd.DataFrame(df_scaled, columns=dataframe.columns)
    
    return df_scaled

if __name__ == "__main__":
    
    df = pd.read_csv('predict.csv',index_col=0)
    df_scaled = scale_data(df)
    
    model = keras.models.load_model('neuralnet.keras')
    
    predictions = model.predict(df_scaled)
    
    print('\n' * 2)
    print('-----------------------------------------------------')
    input('This script converts predict.csv demographic data into a pandas dataframe, loads a pretrained neural net model, makes predictions on total insurance claim cost on that data, and prompts the user for input to display predictions for the chosen record. Press ENTER to continue: ')
    print()
    
    choice = 1
    
    while choice != 0:
        record = int(input("Choose a record # (0-16199): "))
        print()
        print(df.iloc[record])
        print()
        print(f'Predicted total claims amount: ${predictions[record]}')
        print()
        choice = int(input('Type 1 to choose another record or 0 to quit: '))