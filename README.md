# biu_d20_eda
First Project DS Course - Automatic EDA Generator in the Python Language

# General
The project consists of a main.py file, in this file the EDA Generator program is written.
The .env is part of the project and is used to store the relevant API keys used in main.py.
I've added the file to the GitHub project for the sake of the project. 

# Python packages
For the Project to work, it is necessary to install the following packages:
 Pandas, NumPy, 

# The Program
The EDA generator is written in Python and the results are displayed in a Streamlit UI.
The following packages are imported:
 import streamlit as st, import time, import pandas as pd, import requests, import numpy as np, 
 import plotly.graph_objects as go, from dotenv import load_dotenv, import plotly.express as px, 
 from scipy.stats import gaussian_kde, import os

# load UI
in terminal run the following commands: streamlit run main.py

# The UI functionality
First Page - user information
 In the first page the user is asked for a name and city, only then the user gets to see the upload 
 and EDA generator interactive page.

Second Page - interactive EDA generator
 In the second page the user is asked to upload a CSV or EXCEL file, no other file name extension is 
 valid.
 In the sidebar, informative information is displayed: 
  Welcome <User>, 
  Current Time, 
  Weather information from OpenWeatherMap (city entered in first page),
  First three News Headlines from NewsApi,
  email for comments

File Upload
 Invalid file extension: 
  the program displayes an invalid file upload message to load a valid file
 Valid file extension: 
  1. the program checks for an empty file\zero records file
  2. if the file has records then the interactive relevant plots are displayed
  3. the sidebar displayes the different plot sections and gives the user the possibilty to display\not display
     the different sections

  
