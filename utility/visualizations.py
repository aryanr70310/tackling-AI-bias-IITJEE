import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def demographic_pie_chart(dataset, demographic, title):
  # Create pie chart showcasing demographics
  if demographic == "GENDER":
    labels = "Male", "Female"
    explode = [0.1, 0.0]
    colors = ["lightsteelblue", "silver"]
    data = [dataset[dataset["GENDER"]=="M"].shape[0], dataset[dataset["GENDER"]=="F"].shape[0]]
  elif demographic == "CATEGORY":
    labels = "GE", "OC", "SC", "ST"
    explode = [0.1, 0.0, 0.0, 0.0]
    colors = ["lightsteelblue", "silver", "lightblue", "gray"]
    data = [dataset[dataset["CATEGORY"]=="GE"].shape[0], dataset[dataset["CATEGORY"]=="OC"].shape[0], dataset[dataset["CATEGORY"]=="SC"].shape[0], dataset[dataset["CATEGORY"]=="ST"].shape[0]]
  else:
    return

  plt.pie(data, labels = labels, autopct='%1.1f%%', startangle = 15, shadow=True, explode = explode, colors = colors, pctdistance=0.5)
  plt.axis("equal")
  plt.title(title)
  plt.show()

def comparison_bar_plot(compare_df):
  # Generates bar plot for comparing conventional and fair solution results
  # Concept from https://matplotlib.org/2.0.2/examples/api/barchart_demo.html
  n = 6
  ind = np.arange(n)
  width = 0.35
  fig, ax = plt.subplots()

  conventional = compare_df["CONVENTIONAL(%)"]
  rects1 = ax.bar(ind, conventional, width, color="lightblue")

  fair = compare_df["FAIR(%)"]
  rects2 = ax.bar(ind + width, fair, width, color="silver")

  ax.set_ylabel("% of Places in Shortlist")
  ax.set_title("Places in Shortlist by Gender and Category")
  ax.set_xticks(ind + width / 2)
  ax.set_xticklabels(("Male", "Female", "GE", "OC", "SC", "ST"))

  ax.legend((rects1[0], rects2[0]), ("Conventional", "Fair"))

  add_labels(rects1, ax)
  add_labels(rects2, ax)

  plt.show()

def add_labels(rects, ax):
  for rect in rects:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 1.0*height, ('%d' % int(height))+"%", ha="center", va="bottom")