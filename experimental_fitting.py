from scipy.optimize import curve_fit
from scipy.stats import loguniform
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, KFold
from sklearn.base import clone
from sklearn.svm import SVC
from scipy import stats
import os
import statsmodels.api as sm
import pylab as py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

class CurveFit:

    def __init__(self, df):
        self.df = df

        #Making the df easier to work with
        self.df["training_size"] = pd.to_numeric(df["training_size"], downcast="float")
        self.df["accuracy_score"] = pd.to_numeric(df["accuracy_score"], downcast="float")
        self.df = self.df.groupby(["learner", "training_size", "openmlid"]).agg({"accuracy_score" : ['mean', 'std', 'count']})
        self.df.columns = ["average_accuracy_score", 'std', 'count']
        self.df["error"] = 1 - self.df["average_accuracy_score"]
        self.df['average_std'] = self.df['std']
        self.df = self.df.reset_index()
        temp1 = (self.df.groupby(['learner', 'openmlid']).training_size.apply(np.array).reset_index())
        temp2 = (self.df.groupby(['learner', 'openmlid']).error.apply(np.array).reset_index())
        temp3 = (self.df.groupby(['learner', 'openmlid']).average_accuracy_score.apply(np.array).reset_index())
        temp4 = (self.df.groupby(['learner', 'openmlid']).average_std.apply(np.array).reset_index())
        self.df = pd.merge(temp1, temp2, on=['learner', 'openmlid'], how='left')
        self.df = pd.merge(self.df, temp3, on=['learner', 'openmlid'], how='left')
        self.df = pd.merge(self.df, temp4, on=['learner', 'openmlid'], how='left')
        self.df.to_pickle("experiment_results_cleaned.gz")

    def find(self, array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    def exp3(self, x, a, b, c):
        return a*np.exp(-b*x/5000) + c

    def pow3(self, x, a, b, c):
        return a*np.power(x, -b) + c

    def pow2(self,x,a,b):
        return a*np.power(x, -b)

    def log2(self, x, a, c):
        -a*np.log(x) + c

    def exp2(self, x, a, b):
        a*np.exp(-b*x/5000)

    def extrapolate(self, anchor):

        extrapolate_df = []

        for i in range(0, len(self.df)):
            curr_row = self.df.iloc[i]
            arr = curr_row["training_size"]
            value = anchor*arr[len(arr) - 1]
            indx = self.find(curr_row["training_size"], value)
            curr_training = curr_row["training_size"][:indx + 1]
            curr_error = curr_row["error"][:indx + 1]

            try:
                popt,pcov = curve_fit(self.exp3, curr_training, curr_error)

            except (RuntimeError, TypeError):
                popt = [0,-1,0]

            try:
                popt2,pcov2 = curve_fit(self.pow3, curr_training, curr_error, bounds=((0,-2,-3), (5,5,5)))

            except (RuntimeError, TypeError):
                popt2 = [0,-1,0]

            try:
                popt3,pcov3 = curve_fit(self.pow2, curr_training, curr_error)

            except (RuntimeError, TypeError):
                popt3 = [0,-1]

            try:
                popt4,pcov4 = curve_fit(self.log2, curr_training, curr_error)

            except (RuntimeError, TypeError):
                popt4 = [0,1]

            try:
                popt5,pcov5 = curve_fit(self.exp2, curr_training, curr_error)

            except (RuntimeError, TypeError):
                popt5 = [0,1]

            # extrapolate_df.append((curr_row["learner"], curr_row["openmlid"], curr_row["training_size"],curr_row["error"], popt, "exp3", curr_row['average_std']))
            # extrapolate_df.append((curr_row["learner"], curr_row["openmlid"], curr_row["training_size"],curr_row["error"], popt2, "pow3", curr_row['average_std']))
            # extrapolate_df.append((curr_row["learner"], curr_row["openmlid"], curr_row["training_size"],curr_row["error"], popt3, "pow2", curr_row['average_std']))
            # extrapolate_df.append((curr_row["learner"], curr_row["openmlid"], curr_row["training_size"],curr_row["error"], popt4, "log2", curr_row['average_std']))
            extrapolate_df.append((curr_row["learner"], curr_row["openmlid"], curr_row["training_size"],curr_row["error"], popt5, "exp2", curr_row['average_std']))

        df_all_results = pd.DataFrame(np.array(extrapolate_df),
                                      columns=['learner', 'openmlid', 'training_size','error',
                                               'parameters', 'model', 'average_std'])


        df_all_results.to_pickle("extrapolated_parameters.gz")

        new_df = pd.read_pickle("extrapolated_parameters.gz")
        new_df = new_df.sort_values(by = ['openmlid', 'model'])

        for i in range (0,len(new_df) - 1, 2):
            row1 = new_df.iloc[i]
            row2 = new_df.iloc[i+1]
            plt.clf()
            arrx = []
            arry1 = []
            arry2 = []
            training_arr = row1["training_size"]
            max_val = int(training_arr[len(training_arr) - 1])
            model = row1['model']
            parameters1 = row1['parameters']
            parameters2 = row2['parameters']
            for i in range(1, max_val):
                arrx.append(i)
                if model == "exp2":
                    arry1.append(self.exp2(i, *parameters1))
                    arry2.append(self.exp2(i, *parameters2))
                elif model == "exp3":
                    arry1.append(self.exp3(i, *parameters1))
                    arry2.append(self.exp3(i, *parameters2))

                elif model == "pow2":
                    arry1.append(self.pow2(i, *parameters1))
                    arry2.append(self.pow2(i, *parameters2))

                elif model == "pow3":
                    arry1.append(self.pow3(i, *parameters1))
                    arry2.append(self.pow3(i, *parameters2))
                elif model == "log2":
                    arry1.append(self.log2(i, *parameters1))
                    arry2.append(self.log2(i, *parameters2))

            #Extrapolation
            # plt.plot(arrx, arry1,label = (str(row1["learner"])) + str(row1["openmlid"]) + "  extrapolated at " + str(anchor))
            # plt.plot(arrx, arry2,label = (str(row2["learner"])) + str(row2["openmlid"]) + "  extrapolated at " + str(anchor))

            ##Learning curve with standard error
            plt.errorbar(row1["training_size"],row1["error"], marker = ".", yerr = row1['average_std'], linestyle = "None" , label = (str(row1["learner"]) + str(row1["openmlid"])))
            plt.errorbar(row2["training_size"],row2["error"], marker = ".", yerr = row2['average_std'],  linestyle = "None", label = (str(row2["learner"]) + str(row2["openmlid"])))
            # #General learning curve
            # plt.errorbar(row1["training_size"],row1["error"], marker = ".", linestyle = "None", label = (str(row1["learner"]) + str(row1["openmlid"])))
            # plt.errorbar(row2["training_size"],row2["error"], marker = ".",  linestyle = "None", label = (str(row2["learner"]) + str(row2["openmlid"])))
            plt.xlabel("Training size")
            plt.ylabel("Error")
            plt.title("default vs tuned " + str(row1["learner"]) + " on dataset " + str(row1["openmlid"]))
            #plt.legend()
            #Difference
            # row3 = (row1["error"] - row2["error"])/row1["error"]
            # rowerr = row1['average_std'] + row2['average_std']
            # plt.errorbar(row2["training_size"],row3, marker = ".", yerr = rowerr,  linestyle = "None")

            if not os.path.exists("report_images/" + str(row1["openmlid"])):
                os.makedirs("report_images/" + str(row1["openmlid"]))

            if not os.path.exists("report_images/" + str(row1["openmlid"]) + "/curve"):
                os.makedirs("report_images/" + str(row1["openmlid"]) + "/curve")

            plt.savefig("report_images/" + str(row1["openmlid"]) + "/curve" + str(row1['learner']) + ".png")



            #(plt.savefig('HPC_run2/' + str(row1["openmlid"]) + "/curve/" + str(row1['learner']) + str(row1['model']) + "- " + str(int(anchor*10)) + '.png'))
            #(plt.savefig('report_images/images/' + str(row1["openmlid"]) + str(row1['learner'])))
            #(plt.savefig('report_images/errorimages/' + str(row1["openmlid"]) + str(row1['learner'])))

        #(plt.savefig('HPC_run2/' + str(row1["openmlid"]) + "/curve/" + str(row1['learner']) + str(row1['model']) + "- " + str(int(anchor*10)) + 'differencetemp.png'))

















# file_name = 'book3.xlsx'
# test_df = pd.read_pickle("data/experiment_results6.gz")
# # saving the excelsheet
# test_df.to_excel(file_name)
# if not os.path.exists("sus/" + str(row1["openmlid"]) + "std"):
            #     os.makedirs("sus/" + str(row1["openmlid"]) + "std")
            #
            # (plt.savefig('sus/' + str(row1["openmlid"]) + "std/" + str(row1['learner']) + str(row1['model']) + "- " + str(int(anchor*10)) + '.png'))



# if not os.path.exists("plot/" + str(row1["openmlid"])):
#     os.makedirs("plot/" + str(row1["openmlid"]))
#
# (plt.savefig('plot/' + str(row1["openmlid"]) + "/" + str(row1['learner']) + str(row1['model']) + "- " + str(int(anchor*10)) + '.png'))
#(plt.savefig('sus/' + str(row1["openmlid"]) + "/" + str(row1['learner']) + str(row1['model']) + "- " + str(int(anchor*10)) + 'std.png'))




