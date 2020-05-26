#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 01:01:27 2020

@author: William He
Data as of 4/25/2019 from WHO 
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dateutil import parser
%matplotlib qt

url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
dataset = pd.read_csv(url,error_bad_lines=False) 
del url
percent_increase_new = pd.DataFrame([dataset.new_cases/dataset.total_cases]).T
dataset.insert(6,"new_cases_increase", percent_increase_new)
percent_increase_new_death = pd.DataFrame([dataset.new_deaths/dataset.new_cases]).T
dataset.insert(8,"death_rate", percent_increase_new_death)

#Line Plot
def countrydata(name = 'World', x = 'date', y = 'new_cases_increase'):
    country = dataset.loc[dataset.location == name, :]

    fig, ax = plt.subplots()
    #ax.plot(world.date, world.total_deaths)
    ax.plot(country[x],country[y])
    major_ticks = np.arange(min(range(len(country.date))),
                            max(range(len(country.date))),
                            15)
    ax.set_xticks(major_ticks)
    plt.xticks(rotation = 45)
    plt.xlabel(x)
    plt.ylabel(y, rotation = 'vertical')
    plt.title(name)

#Enter Country Name, X_axis, Y_axis to generate Graph
countrydata('United States', 'date', 'death_rate')
countrydata('United States', 'date', 'new_cases_increase')
countrydata('United States', 'date', 'new_cases')


'''
As of 19 May 2020, the columns are: iso_code, location, date, total_cases, new_cases, total_deaths, 
new_deaths, total_cases_per_million, new_cases_per_million, total_deaths_per_million, new_deaths_per_million, 
total_tests, new_tests, new_tests_smoothed, total_tests_per_thousand, new_tests_per_thousand, new_tests_smoothed_per_thousand, 
tests_units, stringency_index, population, population_density, median_age, aged_65_older, aged_70_older, gdp_per_capita, 
extreme_poverty, cvd_death_rate, diabetes_prevalence, female_smokers, male_smokers, handwashing_facilities, hospital_beds_per_100k
'''

filt = ['iso_code','population','total_cases','population_density','median_age','aged_65_older',
        'aged_70_older','gdp_per_capita','cvd_death_rate','diabetes_prevalence','female_smokers',
        'male_smokers','handwashing_facilities','hospital_beds_per_100k']

filt2 = ['population','total_cases','population_density','median_age','aged_65_older',
        'aged_70_older','gdp_per_capita','cvd_death_rate','diabetes_prevalence','female_smokers',
        'male_smokers','handwashing_facilities','hospital_beds_per_100k']
column_name = dataset.loc[:,filt]
filtered = pd.DataFrame(dataset[filt])
filtered.columns = column_name.columns.values
del column_name

#https://stackoverflow.com/questions/19530568/can-pandas-groupby-aggregate-into-a-list-rather-than-sum-mean-etc
d_agg = filtered.groupby('iso_code').aggregate(lambda x: x.unique().tolist())


for num in range (0, len(d_agg.columns)):
    for i in range(0,len(d_agg.population)):
            d_agg.iloc[i,num] = max(d_agg.iloc[i,num])
d_agg = d_agg.drop("OWID_WRL", axis=0)

for names in filt2:
    d_agg[names] = d_agg[names].fillna(-999)
d_agg.replace(-999, np.NaN, inplace=True)

d_agg['InfectRate'] = d_agg.total_cases/d_agg.population * 100

d_agg = d_agg.drop(columns = ['population','total_cases'])
d_agg.dropna(subset=['population_density','gdp_per_capita','cvd_death_rate'], inplace = True)

d_agg['median_age'] = d_agg['median_age'].fillna(np.mean(d_agg.median_age))
d_agg['aged_70_older'] = d_agg['aged_70_older'].fillna(np.mean(d_agg.aged_70_older))
d_agg['aged_65_older'] = d_agg['aged_65_older'].fillna(np.mean(d_agg.aged_65_older))
d_agg['aged_65_older'] = d_agg['aged_65_older'].fillna(np.mean(d_agg.aged_65_older))
d_agg['handwashing_facilities'] = d_agg['handwashing_facilities'].fillna(np.mean(d_agg.handwashing_facilities))

d_agg['male_smokers'] = d_agg['male_smokers'].fillna(np.min(d_agg.male_smokers))
d_agg['female_smokers'] = d_agg['female_smokers'].fillna(np.min(d_agg.female_smokers))
d_agg['hospital_beds_per_100k'] = d_agg['hospital_beds_per_100k'].fillna(np.min(d_agg.hospital_beds_per_100k))


d_agg.info()
len(d_agg.index)


d_agg.corrwith(d_agg.InfectRate).plot.line()
plt.xticks(ticks = range(len(d_agg.columns)-1), labels = ['population_density', 'median_age', 'aged_65_older', 'aged_70_older',
       'gdp_per_capita', 'cvd_death_rate', 'diabetes_prevalence',
       'female_smokers', 'male_smokers', 'handwashing_facilities',
       'hospital_beds_per_100k', 'InfectRate'],rotation = 45)




d_agg.to_csv('cov19_clean.csv', index=True)












