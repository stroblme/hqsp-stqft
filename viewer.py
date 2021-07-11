from tkinter import *
from tkinter.ttk import *
import pickle
from git import exc
import matplotlib
from matplotlib.backend_bases import MouseEvent
import matplotlib.pyplot as plt
from numpy import mat
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
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

    # def show(self, yData, x1Data, title, xlabel, ylabel, x2Data=None, subplot=None, plotType='stem', log=False):
    #     self._show( yData=yData, 
    #                 x1Data=x1Data, 
    #                 title=title, 
    #                 xlabel=xlabel, 
    #                 ylabel=ylabel, 
    #                 x2Data=x2Data, 
    #                 subplot=subplot, 
    #                 plotType=plotType, 
    #                 log=log)
        # # fighandle = plt.figure()

        # if subplot is not None:
        #     plt.subplot(*subplot,frameon=False)
        #     plt.subplots_adjust(wspace=0.58)
        # else:
        #     plt.figure(figsize = (10, 6))

        # fig = plt.gcf()
        # fig.set_size_inches(16,9)
        # fig.canvas.mpl_connect('button_press_event', self.on_click)
        # plt.tight_layout()

        # if x2Data is None:
        #     if log:
        #         ax = plt.gca()
        #         ax.set_yscale('log')
        #         plt.autoscale(False)
        #         plt.ylim(0.1,1)
        #         plt.xlim(min(x1Data), max(x1Data))

        #     if plotType == 'stem':
        #         plt.stem(x1Data, yData)
        #     else:
        #         plt.plot(x1Data, yData, 'o--')


        #     plt.xlabel(xlabel)
        #     plt.ylabel(ylabel)
        # else:
        #     plt.pcolormesh(x2Data, x1Data, yData, cmap=COLORMAP, shading=SHADING)
        #     plt.xlabel(xlabel)
        #     plt.ylabel(ylabel)
                
        # plt.title(title)

        return {'x1Data':x1Data, 'yData':yData, 'x2Data':x2Data, 'subplot':subplot, 'plotType':plotType, 'log':log, 'xlabel':xlabel, 'ylabel':ylabel, 'title':title}

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

            # self.show(yData=yData,x1Data=x1Data,title=title,xlabel=xlabel,ylabel=ylabel,x2Data=x2Data,subplot=subplot,plotType=plotType,log=log)



        fig = plt.gcf()
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

mplv = matplotLibViewer()

mplv.createPlots()

# exit

# def rightKey(event):
#     global pt

#     tpt = pt + 1 if pt < len(fileList)-1 else pt
#     setFigureFromFile(fileList[pt])
#     pt = tpt


# def leftKey(event):
#     global pt

#     tpt = pt - 1 if pt > 0 else pt
#     setFigureFromFile(fileList[pt])
#     pt = tpt


# def resize(event):
#     setFigureFromFile(fileList[pt])


# def startup():
    
#     setFigureFromFile(fileList[pt])

# # w2 = Scale(main, from_=0, to=200, orient=HORIZONTAL)
# # w2.set(23)
# # w2.pack()

# main = Tk()
# frame = Frame(main)
# main.bind('<Left>', leftKey)
# main.bind('<Right>', rightKey)
# # main.bind("<Configure>", resize)


# startup()

# main.mainloop()
