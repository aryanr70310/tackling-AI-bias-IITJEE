from pandas import read_csv, DataFrame, set_option, concat
import numpy as np
import warnings
set_option('mode.chained_assignment', None)

def read_dataset(file_path):
  # Reads dataset at file_path
  names = ["REGST_NO","NAME", "CATEGORY", "GENDER", "MATH", "PHYSICS", "CHEMISTRY"]
  dataset = read_csv(file_path, usecols=names)
  return dataset

def preprocess(dataset, score_coefficient):
  # Preprocess data
  dataset["CATEGORY"] = dataset["CATEGORY"].map({"ON":"GE", "GE":"GE", "SC":"SC", "ST":"ST", "OC":"OC"})

  dataset["MATH"] = dataset["MATH"].astype(float)
  dataset["PHYSICS"] = dataset["PHYSICS"].astype(float)
  dataset["CHEMISTRY"] = dataset["CHEMISTRY"].astype(float)

  dataset["MATH"] += 35
  dataset["PHYSICS"] += 35
  dataset["CHEMISTRY"] += 35

  col_avgs = []
  col_avgs.append(dataset["MATH"].mean())
  col_avgs.append(dataset["PHYSICS"].mean())
  col_avgs.append(dataset["CHEMISTRY"].mean())

  avg_of_subjects = (col_avgs[0] + col_avgs[1] + col_avgs[2])/3
  dataset["MATH"] *= round((avg_of_subjects/col_avgs[0]),2)
  dataset["PHYSICS"] *= round((avg_of_subjects/col_avgs[1]),2)
  dataset["CHEMISTRY"] *= round((avg_of_subjects/col_avgs[2]),2)
  total_marks = (dataset["MATH"] + dataset["PHYSICS"] + dataset["CHEMISTRY"])**score_coefficient

  dataset["TOTAL MARKS"] = total_marks
  dataset.drop("MATH", inplace = True, axis = 1)
  dataset.drop("PHYSICS", inplace = True, axis = 1)
  dataset.drop("CHEMISTRY", inplace = True, axis = 1)

def generate_y(dataset):
  # Create Y column for model
  y_values = len(dataset)*[0]
  y_values = np.array(y_values)
  for i in range(len(dataset),0,-1):
    y_values[len(dataset)-i] = int(100*((i/len(dataset))**1.0))
  dataset["SHORTLIST_CHANCE"] = y_values
  return y_values

def shorten(dataset, cutoff):
  # Cutoff lower scoring candidates
  benchmark = int(dataset.iloc[[int(cutoff*len(dataset))]]["TOTAL MARKS"])
  shortened = dataset[dataset["TOTAL MARKS"] >= benchmark]
  return shortened

def configure_x(X):
  # Prepare X for the model
  X.drop("NAME", inplace=True, axis=1)
  X.drop("SHORTLIST_CHANCE", inplace=True, axis=1)
  X["CATEGORY"] = X["CATEGORY"].map({"GE":0, "SC":1, "OC":2, "ST":3})
  X["GENDER"] = X["GENDER"].map({"M":0, "F":1})
  DataFrame(X).fillna(inplace = True, value = 0)

def diversify(dataset):
  # diversify the dataset naively by randomly removing majority candidates
  dataset.sort_values(by = ["CATEGORY"], inplace = True, ascending = True)
  dataset = dataset[305000:]
  dataset.sort_values(by = ["GENDER"], inplace = True, ascending = False)
  dataset = dataset[41000:]
  return dataset

def reverse_mapping(dataset):
  # Map category and gender to strings
  dataset["CATEGORY"] = dataset["CATEGORY"].map({0:"GE", 1:"SC", 2:"OC", 3:"ST"})
  dataset["GENDER"] = dataset["GENDER"].map({0:"M", 1:"F"})

def devise_shortlist(dataset, cutoff, alpha):
  # shortlist candidates based on predicted score
  shortlist = len(dataset) * ['N']
  shortlist_length = int(0.07*alpha*len(dataset)/cutoff)
  for i in range(0, shortlist_length):
    shortlist[i] = 'Y'
  return shortlist

def devise_rooney_shortlist(dataset, size, alpha):
  # Shortlist algorithm based on Rooney's rule
  warnings.simplefilter(action='ignore', category=UserWarning)

  siz_GE = int(0.07*alpha*len(dataset[dataset["CATEGORY"] == "GE"]))
  df_GE = dataset[dataset["CATEGORY"] == "GE"]
  df_GE["IS_SHORTLISTED"] = devise_shortlist(df_GE, 1.0, alpha) # Allot seats for GE

  siz_OC = int(0.07*alpha*len(dataset[dataset["CATEGORY"] == "OC"])) # Allot seats for OC
  df_OC = dataset[dataset["CATEGORY"] == "OC"]
  df_OC["IS_SHORTLISTED"] = devise_shortlist(df_OC, 1.0, alpha)

  siz_SC = int(0.07*alpha*len(dataset[dataset["CATEGORY"] == "SC"])) # Allot seats for SC
  df_SC = dataset[dataset["CATEGORY"] == "SC"]
  df_SC["IS_SHORTLISTED"] = devise_shortlist(df_SC, 1.0, alpha)

  siz_ST = int(0.07*alpha*len(dataset[dataset["CATEGORY"] == "ST"])) # Allot seats for ST
  df_ST = dataset[dataset["CATEGORY"] == "ST"]
  df_ST["IS_SHORTLISTED"] = devise_shortlist(df_ST, 1.0, alpha)

  dataset = concat([df_GE,df_OC,df_SC,df_ST])

  siz_M = int(0.07*alpha*len(dataset[dataset["GENDER"] == "M"])) - len(dataset[dataset["GENDER"] == "M"][dataset["IS_SHORTLISTED"] == "Y"])
  siz_F = int(0.07*alpha*len(dataset[dataset["GENDER"] == "F"])) - len(dataset[dataset["GENDER"] == "F"][dataset["IS_SHORTLISTED"] == "Y"])

  if siz_M > 0 or siz_F > 0:
    df_M = dataset[dataset["GENDER"] == "M"]
    df_F = dataset[dataset["GENDER"] == "F"]
    if siz_M>0:
      df_M.sort_values(by = ["PREDICTED"], inplace = True, ascending = False)
      fill_shortlist(df_M, siz_M) # Allot seats for Males only if seats aren't already alloted
    if siz_F > 0:
      df_F.sort_values(by = ["PREDICTED"], inplace = True, ascending = False)
      fill_shortlist(df_F, siz_F) # Allot seats for Females only if seats aren't already alloted

  dataset = concat([df_M,df_F])
  dataset.sort_values(by = ["PREDICTED"], inplace = True, ascending = False)
  fill_shortlist(dataset, size - len(dataset[dataset["IS_SHORTLISTED"] == "Y"])) # Fill rest of seats based on merit

  return dataset

def fill_shortlist(dataset, size):
  # Helper function for fair shortlist algorithm
  i, shortlist = 0, len(dataset)*["N"]
  while size>0:
    if dataset.iloc[i]["IS_SHORTLISTED"] == "N":
      size -= 1
    shortlist[i] = "Y"
    i+=1
  dataset["IS_SHORTLISTED"] = shortlist