import pandas as pd
import numpy as np

file = "/Users/Jeremy/python/Ultimate Frisbee/Indoor.csv"
dataset = pd.read_csv(file)


assist_mean = sum(dataset['Assists'])/(len(dataset['Assists'])+.0)
turns_mean = sum(dataset['Turns'])/(len(dataset['Turns'])+.0)
print turns_mean