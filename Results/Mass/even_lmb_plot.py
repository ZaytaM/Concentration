#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 13:39:07 2020

@author: Mathieu Lewin
"""

from numpy import *
from matplotlib.pyplot import plot, figure, title, savefig, subplots
from pickle import dump, load

#plots some curves computed from even_lmb.py
reper='results/'

#################################################
###LMB=6
#################################################
lmb=6
fig1, ax1=subplots(1)

d=3
basefilename='d3_exact_lmb6_a1.1-1.25_Npts20'
A,M,Q = load(open(reper+basefilename+".p","rb"))
ax1.plot(A,M,'b',label='N='+str(d))

d=4
basefilename='d4_exact_lmb6_a0.7-1.25_Npts20'
A,M,Q = load(open(reper+basefilename+".p","rb"))
ax1.plot(A,M,'g',label='N='+str(d))

d=5
basefilename='d5_exact_lmb6_a0.55-1.2_Npts20'
A,M,Q = load(open(reper+basefilename+".p","rb"))
ax1.plot(A,M,'r',label='N='+str(d))

d=6
basefilename='d6_exact_lmb6_a0.5-1.1_Npts30'
A,M,Q = load(open(reper+basefilename+".p","rb"))
ax1.plot(A,M,'m',label='N='+str(d))

xmin=0.5
ymax=2
AA=arange(xmin,1.25,0.01)
YY=arange(0,ymax,0.01)
plot(AA,ones(len(AA)),'k')
plot(ones(len(YY)),YY,'k')
ax1.legend(loc="lower left")
ax1.set_ylim([0,ymax])
fig1.savefig('d3456_exact_lmb6'+'.pdf',bbox_inches='tight')
fig1.savefig('d3456_exact_lmb6'+'.eps',bbox_inches='tight')


#################################################
###LMB=8
#################################################
lmb=8
fig2, ax2=subplots(1)

d=3
basefilename='d3_exact_lmb8_a0.8-1.1_Npts30'
A,M,Q = load(open(reper+basefilename+".p","rb"))
ax2.plot(A,M,'b',label='N='+str(d))

d=4
basefilename='d4_exact_lmb8_a0.5-1.1_Npts30'
A,M,Q = load(open(reper+basefilename+".p","rb"))
ax2.plot(A,M,'g',label='N='+str(d))

d=5
basefilename='d5_exact_lmb8_a0.4-1.1_Npts30'
A,M,Q = load(open(reper+basefilename+".p","rb"))
ax2.plot(A,M,'r',label='N='+str(d))

d=6
basefilename='d6_exact_lmb8_a0.3-1.1_Npts30'
A,M,Q = load(open(reper+basefilename+".p","rb"))
ax2.plot(A,M,'m',label='N='+str(d))

xmin=0.3
ymax=1.5
AA=arange(xmin,1.1,0.01)
YY=arange(0,ymax,0.01)
plot(AA,ones(len(AA)),'k')
plot(ones(len(YY)),YY,'k')
ax2.legend(loc="lower left")
ax2.set_ylim([0,ymax])
fig2.savefig('d3456_exact_lmb8'+'.pdf',bbox_inches='tight')
fig2.savefig('d3456_exact_lmb8'+'.eps',bbox_inches='tight')


#################################################
###LMB=10
#################################################
lmb=10
fig3, ax3=subplots(1)

d=3
basefilename='d3_exact_lmb10_a0.5-1.1_Npts30'
A,M,Q = load(open(reper+basefilename+".p","rb"))
ax3.plot(A,M,'b',label='N='+str(d))

d=4
basefilename='d4_exact_lmb10_a0.3-1.1_Npts30'
A,M,Q = load(open(reper+basefilename+".p","rb"))
ax3.plot(A,M,'g',label='N='+str(d))

d=5
basefilename='d5_exact_lmb10_a0.2-1.1_Npts30'
A,M,Q = load(open(reper+basefilename+".p","rb"))
ax3.plot(A,M,'r',label='N='+str(d))

d=6
basefilename='d6_exact_lmb10_a0.2-0.8_Npts20'
A,M,Q = load(open(reper+basefilename+".p","rb"))
ax3.plot(A,M,'m',label='N='+str(d))

xmin=0.2
ymax=1.5
AA=arange(xmin,1.09,0.01)
YY=arange(0,ymax,0.01)
plot(AA,ones(len(AA)),'k')
plot(ones(len(YY)),YY,'k')
ax3.set_ylim([0,ymax])
ax3.legend(loc="lower left")
fig3.savefig('d3456_exact_lmb10'+'.pdf',bbox_inches='tight')
fig3.savefig('d3456_exact_lmb10'+'.eps',bbox_inches='tight')

#################################################
###LMB=4
#################################################
#here the results have been computed numerically for a check, we do not use the exact formula
lmb=4
fig4, ax4=subplots(1)

d=3
basefilename='d3_exact_lmb4_a1.38-1.5_Npts20'
A,M,Q = load(open(reper+basefilename+".p","rb"))
ax4.plot(A,M,'b',label='N='+str(d))

d=4
basefilename='d4_exact_lmb4_a1.1-1.5_Npts20'
A,M,Q = load(open(reper+basefilename+".p","rb"))
ax4.plot(A,M,'g',label='N='+str(d))

d=5
basefilename='d5_exact_lmb4_a0.9-1.4_Npts20'
A,M,Q = load(open(reper+basefilename+".p","rb"))
ax4.plot(A,M,'r',label='N='+str(d))

d=6
basefilename='d6_exact_lmb4_a0.8-1.3_Npts20'
A,M,Q = load(open(reper+basefilename+".p","rb"))
ax4.plot(A,M,'m',label='N='+str(d))

d=7
basefilename='d7_exact_lmb4_a0.7-1.2_Npts20'
A,M,Q = load(open(reper+basefilename+".p","rb"))
ax4.plot(A,M,'c',label='N='+str(d))

xmin=0.7
ymax=2
AA=arange(xmin,1.5,0.01)
YY=arange(0,ymax,0.01)
plot(AA,ones(len(AA)),'k')
plot(ones(len(YY)),YY,'k')
ax4.legend(loc="lower left")
ax4.set_ylim([0,ymax])
fig4.savefig('d34567_exact_lmb4'+'.pdf',bbox_inches='tight')
fig4.savefig('d34567_exact_lmb4'+'.eps',bbox_inches='tight')

