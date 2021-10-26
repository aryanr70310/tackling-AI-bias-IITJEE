from utility.dataHandler import *
from utility.visualizations import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
import numpy as np
import warnings

dataset = read_dataset("data/jee2009.csv") # read data
scoring_coefficient = 2.9
preprocess(dataset, scoring_coefficient) # preprocess data
print("Data Loaded and Processed")

sub_sample = diversify(dataset) # sub sample diverse set from dataset

# View categorical data in the form of pie charts
# demographic_pie_chart(dataset, "GENDER", "Input Gender Demographics")
# demographic_pie_chart(dataset, "CATEGORY", "Input Category Demographics")

sub_sample.sort_values(by = ["TOTAL MARKS"], inplace = True, ascending = False)
Y = generate_y(sub_sample) # Generate Y column for model
X = sub_sample.copy()
configure_x(X) # Remove strings, characters and NaN
Y = Y[:len(X)]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42) # Split Train and test 75:25
results = X_test.copy()

print("Training Model ...")
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
logistic_regression = LogisticRegression(solver = "lbfgs", max_iter=100)
logistic_regression.fit(X_train, Y_train) # Train logistic regression model
print("Finished training model")

Y_pred = logistic_regression.predict(X_test) # Predict Y for X_test
avg_diff = 0
for i in range(0,len(Y_test)):
  avg_diff += abs(Y_pred[i] - Y_test[i]) # Calculate sum of absolute difference between predicted Y and actual Y
avg_diff/= len(Y_test) # divide by total frequency to retreive an average

results["TOTAL MARKS"] = results["TOTAL MARKS"]**(1/scoring_coefficient)
results["PREDICTED"] = Y_pred

reverse_mapping(results) # Reconvert category and gender to strings

results.sort_values(by = ["PREDICTED"], inplace = True, ascending = False)

range_predicted = results.iloc[0]["PREDICTED"] - results.iloc[len(results)-1]["PREDICTED"] # Max Predicted value - Min Predicted Value
accuracy = 100*(range_predicted - avg_diff)/range_predicted

alpha = 0.6 # constant factored into the rooney-rule-shortlist algorithm
results = devise_rooney_shortlist(results, int(0.07*len(results)), alpha) # Create shortlist based on Rooney's rule
results.sort_values(by = ["PREDICTED","IS_SHORTLISTED", "TOTAL MARKS"], inplace = True, ascending = [False,False,False])

warnings.simplefilter(action='ignore', category=UserWarning)
demographics = []
demographics.append(len(results[results["GENDER"] == "M"][results["IS_SHORTLISTED"] == "Y"]))
demographics.append(len(results[results["GENDER"] == "F"][results["IS_SHORTLISTED"] == "Y"]))
demographics.append(len(results[results["CATEGORY"] == "GE"][results["IS_SHORTLISTED"] == "Y"]))
demographics.append(len(results[results["CATEGORY"] == "OC"][results["IS_SHORTLISTED"] == "Y"]))
demographics.append(len(results[results["CATEGORY"] == "SC"][results["IS_SHORTLISTED"] == "Y"]))
demographics.append(len(results[results["CATEGORY"] == "ST"][results["IS_SHORTLISTED"] == "Y"]))

print("Total number of shortlisted candidates: ",len(results[results["IS_SHORTLISTED"] == "Y"]))
print("Number of Male candidates shortlisted: ",demographics[0])
print("Number of Female candidates shortlisted: ",demographics[1])
print("Number of GE candidates shortlisted: ",demographics[2])
print("Number of OC candidates shortlisted: ",demographics[3])
print("Number of SC candidates shortlisted: ",demographics[4])
print("Number of ST shortlisted candidates: ",demographics[5])

results.to_csv("data/fair.csv") # Write results to csv

print("Test Accuracy(%): {}%".format(accuracy))

# Visualize demographics of shortlisted group
demographic_pie_chart(results[results["IS_SHORTLISTED"] == "Y"], "GENDER", "Shortlist Gender Demographics") 
demographic_pie_chart(results[results["IS_SHORTLISTED"] == "Y"], "CATEGORY", "Shortlist Category Demographics")

demographics = np.array(demographics)
demographics = 100*demographics/len(results[results["IS_SHORTLISTED"] == "Y"])
compare_df = pd.read_csv("data/compare.csv", usecols = ["DEMOGRAPHIC", "CONVENTIONAL(%)"]) # Read shortlist demographics of conventional algorithm
compare_df["FAIR(%)"] = demographics # Add shortlist demographics of fair solution for comparison

comparison_bar_plot(compare_df) # Create bar plot to compare fair and conventional solutions
compare_df.to_csv("data/compare.csv") # Write results to csv file