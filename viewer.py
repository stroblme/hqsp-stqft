# from tkinter import *
# from tkinter.ttk import *
import pickle
import matplotlib.pyplot as plt
# matplotlib.use("TkAgg")
import glob
import os
from numpy.core.fromnumeric import shape
from qbstyles import mpl_style
import numpy as np

from frontend import frontend

frontend.setTheme(dark=False)

cdir = "./data/"
ignoreList = ["venv", ".vscode"]

content = os.listdir(cdir)
folderList = list()

for c in content:
    if os.path.isdir(cdir+c):
        if c not in ignoreList:
            folderList.append(c)

print(f"Found {len(folderList)} folders in current directory:\n {folderList}")


selection = ""
if len(folderList) == 1:
    selection = folderList[0]
else:
    while(selection not in folderList):
        idx = input("Choose the desired datafolder as index (starting from 1)\n")
        # idx=6
        try:
            selection = folderList[int(idx)-1]
        except IndexError:
            continue

    filterString = input("Want to only show a specific plot: tell me here\n")
    print(f"Showing {selection} ...")

fileList = glob.glob(f"{cdir + selection}/*.p")
pt = 0

class matplotLibViewer(frontend):

    def createPlots(self):
        subplots = list()
        plotsDone = list()
        for idx, filePath in enumerate(fileList):
            fileName = os.path.basename(filePath)
            if filterString != "" and filterString not in fileName:
                continue

            if fileName in plotsDone:
                continue

            dataList=list()
            try:
                expN=int(fileName[0])

                subList = [x for i, x in enumerate(fileList) if os.path.basename(x)[0] in "0123456789" and fileName[2:] in x]

                subList.sort()

                for i, item in enumerate(subList):
                    dataList.append(pickle.load(open(item,'rb')))

            except:
                expN=None

                try:
                    dataList.append(pickle.load(open(filePath,'rb')))
                except Exception as e:
                    print(f"Error loading {filePath}: {e}")
                    continue
                
                if "plotdata" not in dataList[0].keys():
                    print(f"Skipping {filePath}")
                    continue
        

            yData = dataList[0]["plotdata"]["yData"]
            x1Data = dataList[0]["plotdata"]["x1Data"]
            title = dataList[0]["plotdata"]["title"]
            xlabel = dataList[0]["plotdata"]["xlabel"]
            ylabel = dataList[0]["plotdata"]["ylabel"]
            x2Data = dataList[0]["plotdata"]["x2Data"]
            subplot = dataList[0]["plotdata"]["subplot"]
            plotType = dataList[0]["plotdata"]["plotType"]
            log = dataList[0]["plotdata"]["log"]

            if subplot in subplots:
                continue

            # if "grader" in fileName:
            #     continue
            subplots.append(subplot)

            if filterString != "":
                subplot = None
                ntitle = input("Enter new title for figure (leave empty to dismiss)\n")
                if ntitle != "":
                    title = ntitle
                    # title = None

            if len(dataList) > 1:
                yDataList = np.zeros(shape=(len(dataList), len(yData)))
                for i, dataItem in enumerate(dataList):
                    yDataList[i][:] = dataItem["plotdata"]["yData"]
                # yDataList=yDataList.transpose()

                plotType="box"
                self._show( yData=yDataList, 
                        x1Data=x1Data, 
                        title=title, 
                        xlabel=xlabel, 
                        ylabel=ylabel, 
                        x2Data=x2Data, 
                        subplot=subplot, 
                        plotType=plotType, 
                        log=log)

                plotsDone.append(fileName)

            else:
                
                self._show( yData=yData, 
                        x1Data=x1Data, 
                        title=title, 
                        xlabel=xlabel, 
                        ylabel=ylabel, 
                        x2Data=x2Data, 
                        subplot=subplot, 
                        plotType=plotType, 
                        log=log)

            if filterString != "":
                break

        fig = plt.gcf()
        fig.canvas.mpl_connect('button_press_event', frontend.on_click)

        plt.show()
        dec = input("Save? [y/n] Sorry need to know that a-priori due to limitations of matplotlib.\n")
        if dec == "y":
            plt.savefig(f'/home/stroblme/Documents/Studium/Semester_12/Masterarbeit/Thesis/figures/{selection}.pdf', format="pdf")

mplv = matplotLibViewer()

mplv.createPlots()

