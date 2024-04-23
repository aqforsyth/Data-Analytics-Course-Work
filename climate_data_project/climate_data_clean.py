#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:21:12 2024

@author: allisonforsyth
"""
import os
os.getcwd()
os.chdir('/Users/allisonforsyth/Documents/GMU_Spring/AIT_580')
os.getcwd()

import pandas as pd


df = pd.read_csv("ICOADS.csv.gz", compression ="gzip",usecols=['STATION','DATE','LONGITUDE','LATITUDE','AIR_TEMP','SEA_SURF_TEMP', 'PAST_WX','PRES_WX','SWELL_DIR','SWELL_HGT','WAVE_HGT','WIND_SPEED', 'WIND_DIR'])
df.head()
max(df["LONGITUDE"])
max(df["LATITUDE"])
min(df["LONGITUDE"])
min(df["LATITUDE"])
#39.724, -77.651, 36.510, -75.608

#filter for chesapeak bay region
df_filter = df[df["LONGITUDE"].between(-77.651, -75.608)]
df_final = df_filter[df_filter["LATITUDE"].between(36.510, 39.724)]
df_final.head()

#date time format
df['date'] = pd.to_datetime(df['DATE'], format="%Y-%m-%dT%H:%M:%S")
df['date'] = df['date'].dt.strftime('%Y/%m/%d')
df.head()

df['date'] = pd.to_datetime(df['date'])
df = df.drop(["DATE"], axis=1)
df.head()

df.to_csv("ICOADS13_23.csv.gzip", index=False, compression="gzip")
df.to_csv("df_climate.csv")