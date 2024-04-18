#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:24:37 2024

@author: allisonforsyth
"""

import os
import plotly.io as pio

pio.renderers.default='browser'

os.getcwd()
os.chdir('/Users/allisonforsyth/Documents/GMU_Spring/CS_504')

import pandas as pd
import plotly.express as px
df = pd.read_csv("nypd.csv")

df[['m', 'd', 'y']] = df['OCCUR_DATE'].str.split('/', expand=True)
df['m'] = df['m'].str.zfill(2)
df['d'] = df['d'].str.zfill(2)
df['date']=df["m"]+df["d"]+df["y"]
df["date"]=pd.to_datetime(df['date'],format='%m%d%y')
df['y'] = df['date'].dt.strftime('%Y')

df= df[["Latitude", "Longitude", "INCIDENT_KEY", "y"]]
df["Latitude"]=df.Latitude.round(2)
df["Longitude"]=df.Longitude.round(2)



mapdf = df.groupby(['Latitude', 'Longitude']).INCIDENT_KEY.agg('count').to_frame('c').reset_index()

fig = px.density_mapbox(mapdf, lat = 'Latitude', lon = 'Longitude', z = "c",
                        radius = 15,
                        center = dict(lat = 40.74, lon = -73.91),
                        zoom = 11,
                        mapbox_style = 'open-street-map')
fig.show()


mapdf_year = df[df["y"]=='2020']
mapdf = mapdf_year.groupby(['Latitude', 'Longitude']).INCIDENT_KEY.agg('count').to_frame('c').reset_index()
fig = px.density_mapbox(mapdf, lat = 'Latitude', lon = 'Longitude', z = "c",
                        radius = 15,
                        center = dict(lat = 40.74, lon = -73.91),
                        zoom = 11,
                        mapbox_style = 'open-street-map')
fig.show()



