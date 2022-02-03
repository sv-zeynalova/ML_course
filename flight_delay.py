import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import streamlit as st
import pydeck as pdk

st.title('FLIGHT_DELAYS')


@st.cache
def make_features(data, max_lag):
    data['year'] = data.DATE.dt.year
    data['month'] = data.DATE.dt.month
    data['day'] = data.DATE.dt.day
    data['dayofweek'] = data.DATE.dt.dayofweek
    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['ARRIVAL_DELAY_x'].shift(lag)


data_load_state = st.text('Loading data...')


@st.cache(allow_output_mutation=True)
def load_data():
    flights = pd.read_csv('Flights3.csv')
    return flights
flights = load_data()
#flights.drop('Unnamed: 0', axis=1, inplace=True)
data_load_state.text("Done!")
st.write(flights.head())


if st.button('В данных есть пропуски, заменим их на средние значения'):
    df_group = flights.groupby(['DESTINATION_AIRPORT', 'AIRLINE'])['ARRIVAL_DELAY'].mean().reset_index().sort_values(
        by='ARRIVAL_DELAY')
    df_all = flights.merge(df_group, how='left', on=['DESTINATION_AIRPORT', 'AIRLINE'])
    df_all.loc[df_all.ARRIVAL_DELAY_x.isna(), 'ARRIVAL_DELAY_x'] = df_all.ARRIVAL_DELAY_y
    del df_all['ARRIVAL_DELAY_y']
    st.write('Пропусков нет: ' + str(df_all.ARRIVAL_DELAY_x.isna().sum()))

    flight_grouped = df_all.groupby(['DATE', 'DESTINATION_AIRPORT'])['ARRIVAL_DELAY_x'].mean().reset_index()
    flight_grouped['DATE'] = pd.to_datetime(flight_grouped['DATE'])
    flight_grouped = pd.DataFrame(flight_grouped)
    st.write('Выполнена группировка для предсказания')

    where_to_go = []
    for dest in flight_grouped.DESTINATION_AIRPORT.unique():
        tempo = flight_grouped[flight_grouped.DESTINATION_AIRPORT == dest][['DATE', 'ARRIVAL_DELAY_x']]
        tempo.columns = ['DATE', 'ARRIVAL_DELAY_x']
        tempo = pd.DataFrame(tempo)
        try:
            make_features(tempo, 21, 7)
            tempo.dropna(inplace=True)
            tempo.set_index('DATE', inplace=True)

            X_train, X_test, y_train, y_test = train_test_split(tempo.drop('ARRIVAL_DELAY_x', axis=1),tempo.ARRIVAL_DELAY_x, shuffle=False,test_size=0.25)

            model_lr = LinearRegression()
            model_lr.fit(X_train, y_train)

            y_predicted_lr = model_lr.predict(X_test)
            where_to_go.append([dest, y_test.mean(), np.sqrt(mean_squared_error(y_test, y_predicted_lr))])

        except Exception as e:
            print('Error', str(e))
    st.write('Модель обучена и предсказания выполнены')

    where_to_go = pd.DataFrame(where_to_go)
    st.write(where_to_go.head())
    where_to_go.columns = ['DESTINATION_AIRPORT', 'MEAN_ARRIVAL_DELAY_PREDICT', 'RMSE']


if st.button('Выберите аэропорт вылета:'):
    ad=st.multiselect(label='Aэропорт',options=flights['ORIGIN_AIRPORT'].unique())

    start_airport = ad
    where_to_go_from_start_airport = flights[flights.ORIGIN_AIRPORT == start_airport]['DESTINATION_AIRPORT'].unique()

    where_to_go_from_start_airport = pd.DataFrame(where_to_go_from_start_airport)
    where_to_go_from_start_airport.columns = ['DESTINATION_AIRPORT']

    top3 = where_to_go_from_start_airport \
        .merge(where_to_go, on='DESTINATION_AIRPORT', how='inner') \
        .sort_values(by=['RMSE', 'MEAN_ARRIVAL_DELAY_PREDICT'], ascending=[True, True]) \
        .head(3)

    print('Лучшие направления с аэропорта', start_airport)
    print('')
    print(top3)
