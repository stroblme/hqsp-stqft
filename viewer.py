from tkinter import *
from tkinter.ttk import *
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import glob
import os
from qbstyles import mpl_style

from frontend import frontend

COLORMAP = 'plasma'
SHADING='nearest'
DARK=True

mpl_style(dark=DARK, minor_ticks=False)

cdir = "./data"
ignoreList = ["venv", ".vscode"]

content = os.listdir(cdir)
folderList = list()

for c in content:
    if os.path.isdir(c):
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

fileList = glob.glob(f"{cdir + selection}/*.p")
pt = 0

class matplotLibViewer(frontend):

    clickEventHandled = True

    def on_click(self, event):
        """Enlarge or restore the selected axis."""
        self.clickEventHandled

        if not self.clickEventHandled:
            return

        ax = event.inaxes
        if ax is not None:
            # Occurs when a region not in an axis is clicked...
            if int(event.button) is 1:
                # On left click, zoom the selected axes
                ax._orig_position = ax.get_position()
                ax.set_position([0.1, 0.1, 0.85, 0.85])
                for axis in event.canvas.figure.axes:
                    # Hide all the other axes...
                    if axis is not ax:
                        axis.set_visible(False)
                event.canvas.draw()

            elif int(event.button) is 3:
                # On right click, restore the axes
                try:
                    ax.set_position(ax._orig_position)
                    for axis in event.canvas.figure.axes:
                        axis.set_visible(True)
                except AttributeError:
                    # If we haven't zoomed, ignore...
                    pass

                event.canvas.draw()

        self.clickEventHandled = True

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
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

mplv = matplotLibViewer()

mplv.createPlots()

