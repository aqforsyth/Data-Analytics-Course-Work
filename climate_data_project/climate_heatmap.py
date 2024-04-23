#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 21:34:19 2024

@author: allisonforsyth
"""

import os
import plotly.io as pio

pio.renderers.default='browser'

os.getcwd()
os.chdir('/Users/allisonforsyth/Documents/GMU_Spring/AIT_580')

import pandas as pd
import plotly.express as px
df = pd.read_csv("airpercentchange.csv")
df

df["Percent Change in Air Temperature"] = df["yoy_per_change_at"]


fig = px.density_mapbox(df, lat = 'LATITUDE', lon = 'LONGITUDE', z = "Percent Change in Air Temperature",
                        radius = 40,
                        center = dict(lat = 38, lon = -75.50),
                        zoom = 7,
                        mapbox_style = 'open-street-map')
fig.show()

df2 = pd.read_csv("seapercentchange.csv")
df2

df2["Percent Change in Sea Surface Temperature"] = df2["yoy_per_change_sst"]

fig = px.density_mapbox(df2, lat = 'LATITUDE', lon = 'LONGITUDE', z = "Percent Change in Sea Surface Temperature",
                        radius = 40,
                        center = dict(lat = 38, lon = -75.50),
                        zoom = 7,
                        mapbox_style = 'open-street-map')
fig.show()