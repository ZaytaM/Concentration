#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 09:50:10 2020

@author: Mathieu Lewin

Edited by Matias G. Delgadino
"""
from numpy import *
#import matplotlib
#matplotlib.use('Agg')
from matplotlib.pyplot import plot, figure, title, savefig, subplots
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import gamma
from pickle import dump, load

# Case lmb=2*k with k>1 in the reversed Hardy-Littlewood-Sobolev inequality of
# Carillo, Delgadino, Dolbeault, Frank & Hoffmann, J. Math. Pures Appl. (2019)
# Computes the nonlinear solution the mass of the solution to the Euler-Lagrange equation, for a fixed Lagrange multiplier L


##############################################################
##############################################################
####PARAMETERS TO BE DEFINED FOR THE COMPUTATION
Npert=10 # Nb of random perturbations to try for finding the solution
use_nonuniform_grid=True #use of a non-uniform grid or not (might give slightly better results when there is a delta)
be_silent=False #whether to print more info or not
save_fig=True #whether to save the results on the disk
##############################################################
##############################################################

#area of the hypersphere S^(d-1) in R^d
def sphere(d):
    return 2*pi**(d/2)/gamma(d/2)

#computes q in terms of a for lmb=8
def q(d,a,lmb):
    return d*(2-a)/(d*(2-a)+lmb)

#critical alpha below which there cannot be condensation (the mass diverge)
def a_crit(d,lmb):
    return max(0,2-lmb/2+lmb/d)

#coefficients from the double integral
def coeff(d,lmb):
    if lmb==6:
        return [3+12/d,3+12/d]
    elif lmb==8:
        return [(32+48/d+4*d)/(2+d), (60+144/d+6*d)/(2+d), (32+48/d+4*d)/(2+d)]
    elif lmb==10:
        return [(50+80/d+5*d)/(2+d), (140+480/d+10*d)/(2+d), (140+480/d+10*d)/(2+d), (50+80/d+5*d)/(2+d)]
    else: #lmb==4
        return [2+4/d]

#density multiplied by $x^{2/(1-q)}$
def RhoxR(r,C,L,d,qq,lmb):
    cf=coeff(d,lmb)
    Rho= r**(-2/(1-qq))*(L+r**(lmb)+ sum([C[i]**2*cf[i]*r**(lmb-2*i-2) for i in range(int(lmb/2)-1)]))**(-1/(1-qq))
    return (qq/(1-qq))**(1/(1-qq))*Rho

#the integral used all the times \int_0^\infty r^\alpha(x^lmb+...)^{-\beta}dr
def IntRef(C,L,d,lmb,alpha,beta):
    cf=coeff(d,lmb)
    fun=lambda r: r**alpha*(L+r**(lmb) + sum([C[i]**2*cf[i]*r**(lmb-2*i-2) for i in range(int(lmb/2)-1)]))**(-beta)
    return quad(fun,0,+inf)[0]

#\int_{R^d} |x|^n Rho
def Mn(C,L,d,qq,lmb,n):
    return sphere(d)*(qq/(1-qq))**(1/(1-qq))*IntRef(C,L,d,lmb,d-1+n,1/(1-qq))

#mass of Rho = Mn(...,0)
def mass(C,L,d,qq,lmb):
    return sphere(d)*(qq/(1-qq))**(1/(1-qq))*IntRef(C,L,d,lmb,d-1,1/(1-qq))

#function to be minimized (minimum is 0 for the solution), with C=[A,B]
def fn(C,L,d,qq,lmb):
    return sum([(Mn(C,L,d,qq,lmb,2*j+2)-C[j]**2)**2 for j in range(int(lmb/2)-1)])

#its gradient
def fn_grad(C,L,d,qq,lmb):
    cf=coeff(d,lmb)
    gf=zeros(int(lmb/2)-1)
    for j in range(int(lmb/2)-1):
        dIntj=array([IntRef(C,L,d,lmb,d-1+2*j+2+lmb-2*i-4,1/(1-qq)+1) for i in range(int(lmb/2)-1)])
        dMnj=-2/(1-qq)*dIntj*C*cf*sphere(d)*(qq/(1-qq))**(1/(1-qq))
        dCj=zeros(int(lmb/2)-1)
        dCj[j]=2*C[j]
        gf=gf+2*(Mn(C,L,d,qq,lmb,2*j+2)-C[j]**2)*(dMnj-dCj)
    return gf

#minimization with initial C_init
def solve_AB(C_init,L,d,qq,lmb):
    res=minimize(fn,C_init,args=(L,d,qq,lmb),method='BFGS',jac=fn_grad,tol=1e-10)
    return res.x,res.fun

#finds the fixed point solution by running solve_AB from Npert random C_init. Returns the mass of the best one
def find_mass(L,d,qq,lmb):
    #finds minimum starting from A=1 and B=1
    lenC=int(lmb/2)-1
    C_init=10*ones(lenC)
    Copt,fopt=solve_AB(C_init,L,d,qq,lmb)
    #finds minimum starting from random initial conditions A,B and take the best
    for _ in range(Npert):
        C_init=random.rand(lenC)
        C2,f2=solve_AB(C_init,L,d,qq,lmb)
        if f2<fopt:
            Copt,fopt=C2,f2
    mopt=mass(Copt,L,d,qq,lmb)
    #prints the result
    if be_silent==False:
        print('At q=',qq,'and L=',L,'found',Copt,'with error=',sqrt(fopt))
        print('Mass=',mopt)
    return mopt,Copt,fopt

#main function to plot the mass of the fixed point solution on [amin,amax] with Na points
def curve_mass(Lmin,Lmax,d,lmb,qmin,qmax,Nq,NL):
    print('Computing the mass of the solution with a given Langrange multiplier',d,'d at lmb=',lmb,'...')
#    print('Parameters are: N=',N,'Rmax=',Rmax,'Adapted grid=',str(use_nonuniform_grid))
    Q=arange(qmin,qmax,(qmax-qmin)/Nq)
    Larray=arange(Lmin,Lmax,(Lmax-Lmin)/NL)
    M=zeros((len(Q),len(Larray)))
    E=zeros((len(Q),len(Larray)))

    fig, axes = subplots(nrows=len(Q), ncols=1, sharex=True, sharey=False, figsize=(20,20))
    for j in range(len(Q)):
      for i in range(len(Larray)):
          mopt,Copt,fopt=find_mass(Larray[i],d,Q[j],lmb)
          M[j,i]=mopt
          E[j,i]=fopt
    # plots the mass  in terms of L for diferent values of q
      axes[j].plot(Larray,M[j],'b')
    #saves everything on the disk
    if save_fig==True:
        basefilename='d'+str(d)+'_lmb'+str(lmb)+'_q'+str(qmin)+'-'+str(qmax)+'_Npts'+str(Nq)+'_L'+str(Lmin)+'-'+str(Lmax)+'_NL'+str(NL)
        fig.savefig(basefilename+'_LM'+'.pdf')
    #     dump([Larray,M,Q], open(basefilename+'.p', "wb"))
    return Larray,M,Q,E

##########################################################
####MAIN PROGRAM TO BE RUN
##########################################################
Nq=10 #number of points to compute
Lmin=0.01
Lmax=5
NL=10
tol=0.1


for d in [4,5,6,7,8,9,10]:
  for lmb in [6,8,10]:
    qmin=d/(d+lmb) #minimal value of a for the plot
    qmax=1 #maximal value of a for the plot
    Larray,M,Q,E=curve_mass(Lmin,Lmax,d,lmb,qmin,qmax,Nq,NL) #plots the curve
    m=-100
    for k in range(Nq):
      for j in range (NL-1):
        m1=M[k,j+1]-M[k,j]
        e=max(E[k,j+1],E[k,j])
      if m1>m:
        if e<tol:
          m=M[k,j+1]-M[k,j]
    with open("derivative1.txt",'a') as arq:
      arq.write('\n')
      arq.write('d'+str(d)+'_lmb'+str(lmb)+'_q'+str(qmin)+'-'+str(qmax)+'_Npts'+str(Nq)+'_L'+str(Lmin)+'-'+str(Lmax)+'_NL'+str(NL))
      arq.write('\n')
      arq.write(str(m))
