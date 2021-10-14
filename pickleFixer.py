import pickle
import glob
import os

def update_details(filePath):
    
    data = pickle.load(open(filePath,'rb'))
    # subplot shift
    # if "qft_real" not in data['identifier'] or 'grader' in data['identifier']:
        # return
    # data['plotdata']['subplot'][2] = data['plotdata']['subplot'][2]+1

    if 'grader' not in data['identifier']:
        return
    data['plotdata']['x1Data']=data['plotdata']['x1Data'][-5:]
    data['plotdata']['yData']=data['plotdata']['yData'][-5:]

    pickle.dump(data, open(filePath, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
  
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

fileList = glob.glob(f"{cdir + selection}/*.p")

for file in fileList:

    update_details(file)

print("done")