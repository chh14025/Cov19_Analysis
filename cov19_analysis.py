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
%matplotlib inline

dataset = pd.read_csv('owid-covid-data.csv')
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
This analysis is quite useless as the plot legend covers every single country
making it quite difficult to read.
POTENTIAL SOLUTION: sort countries by regions

countryset = pd.DataFrame(dataset)
countryset.dropna(subset = ['iso_code'], inplace=True)
dataset.tail()
countryset.tail()

g = sns.lmplot(data = countryset, x= 'new_deaths', y = 'new_cases',fit_reg = False, hue = 'location')
g._legend.remove()
'''


testset = pd.DataFrame(dataset)
testset.dropna(subset = ['total_tests'], inplace=True)



















