
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys

##=============================================##
##=============================================##
def is_comment(s):
    """ function to check if a line
         starts with some character.
         Here # for comment
    """
    # return true if a line starts with #
    return s.startswith('ITEM: NUMBER OF ATOMS')
##=============================================##
##=============================================##
def readLog(filename):
    tStep = 0
    readSwitch = 0
    numSwitch = 0
    step = []
    output = []
    with open(filename,'r') as f:
        for line in f.readlines():
            # print(readSwitch,line)
            if line.startswith("Loop time"):
                if readSwitch == 1:
                    output.append(np.array(step))
                    readSwitch = 0

            if readSwitch == 1:
                line = line.strip()
                # line = line.split(' ')
                buf = []
                for number in line.split():
                    buf.append(number)
                buf=np.array(buf,dtype = float)
                step.append(buf)

            if line.startswith("Step Temp"):
                readSwitch = 1
                numSwitch += 1

                step = []


    return np.array(output)

##=============================================##
##=============================================##

def readLigands(filename):
    tStep = 0
    readSwitch = 0
    numSwitch = 0
    step = []
    output = []
    with open(filename,'r') as f:
        for line in f.readlines():
            # print(readSwitch,line)
            if line.startswith("pair_modify		pair 	lj/cut	shift yes"):
                if readSwitch == 1:
                    output.append(np.array(step))
                    readSwitch = 0

            if readSwitch == 1:
                line = line.strip()
                # line = line.split(' ')
                # print(line)
                buf = []
                for number in line.split():
                    buf.append(number)
                buf=np.array(buf,dtype = object)
                step.append(buf)

            if line.startswith("#membrane-np interactions"):
                readSwitch = 1
                numSwitch += 1

                step = []


    return np.array(step)

##=============================================##
##=============================================##
def readDump(filename):
    tStep = 0
    readSwitch = 0
    numSwitch = 0
    step = []
    output = []
    with open(filename,'r') as f:
        for line in f.readlines():
            # print(readSwitch,line)
            if line.startswith("ITEM: TIMESTEP"):
                if readSwitch == 1:
                    output.append(np.array(step))
                    readSwitch = 0

            if readSwitch == 1:
                line = line.strip()
                # line = line.split(' ')
                buf = []
                for number in line.split():
                    buf.append(number)
                buf=np.array(buf,dtype = float)
                step.append(buf)

            if line.startswith("ITEM: ATOMS"):
                readSwitch = 1
                numSwitch += 1

                step = []
    return np.array(output)
##=============================================##
##=============================================##

##=============================================##
##=============================================##
def splitDump(filename): ## split dump into time steps
    workingData = np.array(pd.read_csv(filename,sep = "\s+", names=forceFormat, header = None, dtype=object))
    tStep = 0
    readSwitch = 0
    numSwitch = 0
    step = []
    output = []

    for i in range(int(workingData.shape[0])):


        if workingData[i,1]=='TIMESTEP':

            if readSwitch == 1:
                output.append(np.array(step))
                readSwitch = 0

        if readSwitch == 1:
            step.append(workingData[i])


        if workingData[i,1]=='ATOMS':

            readSwitch = 1
            numSwitch += 1

            step = []

    return np.array(output, dtype = float)
##=============================================##
##=============================================##

##=============================================##
def splitMovie(filename, movieFormat): ## split dump into time steps
    workingData = np.array(pd.read_csv(filename,sep = "\s+", names=movieFormat, header = None, dtype=object))
    linesToPurge = 9 ## works with custom dump of lammps
    numParts = int(workingData[3,0]) ## this function only works for conserved number of particles
    numSplit = int(len(workingData)/(numParts + linesToPurge ))
    output = np.reshape( workingData, (numSplit,numParts+linesToPurge,workingData.shape[1]))
    outputCut = output[:,linesToPurge:,:]
    assert(outputCut.shape[1] == numParts)
    return  np.array(outputCut, dtype = float)
##=============================================##
##=============================================##

def splitMovie_too_slow(filename, movieFormat):
    # counter = 0
    # ## first determine number of particles from LAMMPS dump, then fill numpy array with corresponding data
    #
    # num_parts = 0
    # num_columns = 0
    #
    # with open(filename, 'r') as f: ## pre load first line to determine the size of the output array
    #     for line in f.readlines():
    #         line_buffer = line.strip().split()
    #
    #         if counter == 0: ## some checks on the file format
    #             if line.startswith("ITEM: TIMESTEP"):
    #                 pass
    #             else:
    #                 print("ERROR: Ooops wrong movie format!")
    #                 sys.exit()
    #         if counter == 2:
    #
    #             if line.startswith("ITEM: NUMBER OF ATOMS"):
    #                 pass
    #             else:
    #                 print("ERROR: Ooops wrong movie format!")
    #                 sys.exit()
    #         if counter == 4:
    #
    #             if line.startswith("ITEM: BOX BOUNDS pp pp ff"):
    #                 pass
    #             else:
    #                 print("ERROR: Ooops wrong movie format!")
    #                 sys.exit()
    #
    #         if counter == 3:
    #             num_parts = int(line_buffer[0])
    #         if counter == 9:
    #             num_columns = len(line_buffer)
    #
    #         counter += 1
    #
    # print(counter)

    tStep = 0
    readSwitch = 0
    numSwitch = 0
    step = []
    output = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            # print(readSwitch,line)
            if line.startswith("ITEM: TIMESTEP"):
                if readSwitch == 1:
                    output.append(np.array(step))
                    readSwitch = 0

            if readSwitch == 1:
                line = line.strip()
                # line = line.split(' ')
                buf = []
                for number in line.split():
                    buf.append(number)
                buf = np.array(buf, dtype=float)
                step.append(buf)

            if line.startswith("ITEM: ATOMS"):
                readSwitch = 1
                numSwitch += 1

                step = []
    return np.array(output)

