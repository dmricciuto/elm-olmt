#!/usr/bin/env python
import sys,os, time
import numpy as np
import subprocess
import pickle
import model_ELM
from optparse import OptionParser

parser = OptionParser()

parser.add_option("--case", dest="case", default="", \
                  help="Case name")
parser.add_option("--cases_compare", dest="cases_compare", default="", \
                  help="Additional cases to plot for comparison")
parser.add_option("--plot_spinup", dest="spinup", default=False, \
                  action="store_true")
(options, args) = parser.parse_args()

#Post-process and plot variables (non-ensemble mode)

#Load case object
myfile=open('pklfiles/'+options.case+'.pkl','rb')
mycase = pickle.load(myfile)

#mycase.postproc_vars = 
#mycase.postproc_startyear = 
#mycase.postproc_endyear = 
#mycase.postproc_freq =

if options.spinup:
    mycase.plot_spinup()
else:
    for v in mycase.postproc_vars:
        if ('_pft' in v):
            hnum = 2 #Assume pft-level output is in h2 file
            myindex = list(mycase.postproc_pfts)
        else:
            hnum = 1
            myindex = [0]
        #Note, we assume daily outputs.
        annualmean = False
        dailytomonthly = False
        meanseasonalcycle = False
        if (mycase.postproc_freq.lower() == 'monthly'):
            dailytomonthly = True
        elif (mycase.postproc_freq.lower() == 'annual'):
            annualmean = True
        for i in myindex:
            mycase.postprocess(v, index=i, gindex=0, startyear=mycase.postproc_startyear, \
                endyear=mycase.postproc_endyear, hnum=hnum, \
                dailytomonthly=dailytomonthly, annualmean=annualmean,  \
                meanseasonalcycle=meanseasonalcycle, xindex=0,yindex=0, plot=True)

