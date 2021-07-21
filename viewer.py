# from tkinter import *
# from tkinter.ttk import *
import pickle
import matplotlib.pyplot as plt
# matplotlib.use("TkAgg")
import glob
import os
from qbstyles import mpl_style

from frontend import frontend

frontend.setTheme(dark=True)

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
        try:
            selection = folderList[int(idx)-1]
        except IndexError:
            continue

    print(f"Showing {selection} ...")

fileList = glob.glob(f"{cdir + selection}/*.p")
pt = 0

class matplotLibViewer(frontend):

    def createPlots(self):
        for filePath in fileList:
            try:
                data = pickle.load(open(filePath,'rb'))
            except Exception as e:
                print(f"Error loading {filePath}: {e}")
                continue
            
            if "plotdata" not in data.keys():
                print(f"Skipping {filePath}")
                continue
        

            yData = data["plotdata"]["yData"]
            x1Data = data["plotdata"]["x1Data"]
            title = data["plotdata"]["title"]
            xlabel = data["plotdata"]["xlabel"]
            ylabel = data["plotdata"]["ylabel"]
            x2Data = data["plotdata"]["x2Data"]
            subplot = data["plotdata"]["subplot"]
            plotType = data["plotdata"]["plotType"]
            log = data["plotdata"]["log"]

            self._show( yData=yData, 
                    x1Data=x1Data, 
                    title=title, 
                    xlabel=xlabel, 
                    ylabel=ylabel, 
                    x2Data=x2Data, 
                    subplot=subplot, 
                    plotType=plotType, 
                    log=log)

        fig = plt.gcf()
        fig.canvas.mpl_connect('button_press_event', frontend.on_click)
        plt.show()

mplv = matplotLibViewer()

mplv.createPlots()

