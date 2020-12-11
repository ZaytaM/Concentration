#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Sep 26 09:50:10 2020
@author: Mathieu Lewin
Edited: Matias Delgadino
"""

from numpy import *
from matplotlib.pyplot import plot, figure, title, savefig, subplots
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import gamma
from pickle import dump, load

# Case lmb=2*k with k>1 in the reversed Hardy-Littlewood-Sobolev inequality of
# Carillo, Delgadino, Dolbeault, Frank & Hoffmann, J. Math. Pures Appl. (2019)
# Computes the nonlinear solution to the Euler Lagrange equation for diferent values of the Lagrange multiplier L. 
# Condensation happens when its mass is <1

##############################################################
##############################################################
####PARAMETERS TO BE DEFINED FOR THE COMPUTATION
Npert=0 # Nb of random perturbations to try for finding the solution
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

#critical alpha below which there cannot be condensation (the mass diverge at the origin)
def a_crit(d,lmb):
    return max(0,2-lmb/2+lmb/d)

#critical alpha above which the problem is not defined (corresponds to q=(N-2)/(N+2))
def a_crit2(d,lmb):
    return max(0,2-lmb/4+lmb/2/d)

#coefficients from the double integral
def coeff(d):
    if lmb==6:
        return [3+12/d,3+12/d]
    elif lmb==8:
        return [(32+48/d+4*d)/(2+d), (60+144/d+6*d)/(2+d), (32+48/d+4*d)/(2+d)]
    elif lmb==10:
        return [(50+80/d+5*d)/(2+d), (140+480/d+10*d)/(2+d), (140+480/d+10*d)/(2+d), (50+80/d+5*d)/(2+d)]
    else: #lmb==4
        return [2+4/d]

#density multiplied by $x^{2/(1-q)}$
def RhoxR(L,r,C,d,a,lmb):
    qq=q(d,a,lmb)
    cf=coeff(d)
    Rho= r**(-2/(1-qq))*(r**(lmb-2)+ L + sum([C[i]**2*cf[i]*r**(lmb-2-2*i-2) for i in range(int(lmb/2)-1)]))**(-1/(1-qq))
    return (qq/(1-qq))**(1/(1-qq))*Rho

#the integral used all the times \int_0^\infty r^\alpha(x^lmb+...)^{-\beta}dr
def IntRef(L,C,d,lmb,alpha,beta):
    cf=coeff(d)
    fun=lambda r: r**alpha*(r**(lmb-2)+ L + sum([C[i]**2*cf[i]*r**(lmb-2*i-4) for i in range(int(lmb/2)-1)]))**(-beta)
    return quad(fun,0,+inf)[0]

#\int_{R^d} |x|^n Rho
def Mn(L,C,d,a,lmb,n):
    qq=q(d,a,lmb)
    return sphere(d)*(qq/(1-qq))**(1/(1-qq))*IntRef(L,C,d,lmb,d-1-2/(1-qq)+n,1/(1-qq))

#mass of Rho = Mn(...,0)
def mass(L,C,d,a,lmb):
    qq=q(d,a,lmb)
    return sphere(d)*(qq/(1-qq))**(1/(1-qq))*IntRef(L,C,d,lmb,d-1-2/(1-qq),1/(1-qq))

#function to be minimized (minimum is 0 for the solution), with C=[A,B]
def fn(C,L,d,a,lmb):
    return sum([(Mn(L,C,d,a,lmb,2*j+2)-C[j]**2)**2 for j in range(int(lmb/2)-1)])

#its gradient
def fn_grad(C,L,d,a,lmb):
    cf=coeff(d)
    qq=q(d,a,lmb)
    gf=zeros(int(lmb/2)-1)
    for j in range(int(lmb/2)-1):
        dIntj=array([IntRef(L,C,d,lmb,d-1-2/(1-qq)+2*j+2+lmb-2*i-4,1/(1-qq)+1) for i in range(int(lmb/2)-1)])
        dMnj=-2/(1-qq)*dIntj*C*cf*sphere(d)*(qq/(1-qq))**(1/(1-qq))
        dCj=zeros(int(lmb/2)-1)
        dCj[j]=2*C[j]
        gf=gf+2*(Mn(L,C,d,a,lmb,2*j+2)-C[j]**2)*(dMnj-dCj)
    return gf

#minimization with initial C_init
def solve_AB(C_init,L,d,a,lmb):
    res=minimize(fn,C_init,args=(L,d,a,lmb),method='BFGS',jac=fn_grad,tol=1e-10)
    return res.x,res.fun

#finds the fixed point solution by running solve_AB from Npert random C_init. Returns the mass of the best one
def find_mass(L,d,a,lmb,C_init):
    #finds minimum starting from given initial state
    lenC=int(lmb/2)-1
    Copt,fopt=solve_AB(C_init,L,d,a,lmb)
    #finds minimum starting from random initial conditions A,B and take the best
    for _ in range(Npert):
        C_init=10*random.rand(lenC)
        C2,f2=solve_AB(C_init,L,d,a,lmb)
        if f2<fopt:
            Copt,fopt=C2,f2
    mopt=mass(L,Copt,d,a,lmb)
    #prints the result
    if be_silent==False:
        print('At a=',a,'found',Copt,'with error=',sqrt(fopt))
        print('Mass=',mopt)
    return mopt,Copt,fopt

#function to plot the mass of the fixed point solution on [amin,amax] with Na points
def curve_mass_a(L,d,lmb,amin,amax,Na):
    print('Computing the mass of the simple solution in',d,'d at lmb=',lmb,'...')
    A=arange(amin,amax,(amax-amin)/Na)
    M=zeros(len(A))
    Q=zeros(len(A))
    for i in range(len(A)):
        Q[i]=q(d,A[i],lmb)        
        mopt,Copt,_=find_mass(L,d,A[i],lmb,ones(int(lmb/2)-1))
        M[i]=mopt
    #plots the mass in terms of a and q
    fig1, ax1=subplots(1)
    ax1.plot(A,M,'b')
    ax1.plot(A,ones(len(A)),'k')
    fig2, ax2=subplots(1)
    ax2.plot(Q,M,'b')
    ax2.plot(Q,ones(len(A)),'k')
    #saves everything on the disk
    if save_fig==True: 
        basefilename='d'+str(d)+'_exact_lmb'+str(lmb)+'_a'+str(amin)+'-'+str(amax)+'_Npts'+str(Na)
        fig1.savefig(basefilename+'_AM'+'.pdf')
        fig2.savefig(basefilename+'_QM'+'.pdf')
        dump([A,M,Q], open(basefilename+'.p', "wb"))        
    return A,M,Q

#Plot the mass of the fixed point solution on [Lmin,Lmax] with NL points for q ranging in [qmin,qmax] with Nq points
def curve_mass_L(Lmin,Lmax,d,lmb,amin,amax,Na,NL):
    print('Computing the mass of the solution with a given Langrange multiplier',d,'d at lmb=',lmb,'...')
    A=arange(amin,amax,(amax-amin)/Na)
    Q=zeros(len(A))
    Larray=arange(Lmin,Lmax,(Lmax-Lmin)/NL)
    M=zeros((len(A),len(Larray)))
    E=zeros((len(A),len(Larray)))

    fig, axes = subplots(nrows=len(Q), ncols=1, sharex=True, sharey=False, figsize=(20,20))
    for j in range(len(A)):
      Q[j]=q(d,A[j],lmb)
      for i in range(len(Larray)):
          mopt,Copt,fopt=find_mass(Larray[i],d,A[j],lmb,ones(int(lmb/2)-1))
          M[j,i]=mopt
          E[j,i]=fopt
    # plots the mass  in terms of L for diferent values of q
      axes[j].plot(Larray,M[j],'b')
    #saves everything on the disk
    if save_fig==True:
        basefilename='d'+str(d)+'_lmb'+str(lmb)+'_q'+str(qmin)+'-'+str(qmax)+'_Npts'+str(Nq)+'_L'+str(Lmin)+'-'+str(Lmax)+'_NL'+str(NL)
        fig.savefig(basefilename+'_LM'+'.pdf')
    return Larray,M,Q,E
    
##########################################################
####MAIN PROGRAM TO BE RUN
##########################################################
d=5 #space dimension
lmb=8

# ##option 1: find the mass for just one a
# #a=0.9
# #find_mass(d,a,lmb)

##option 2: find the mass for many a's and plot the curve
amin=0.6 #minimal value of a for the plot
amax=1.1 #maximal value of a for the plot
Na=20 #number of points to compute
A,M,Q=curve_mass_a(0,d,lmb,amin,amax,Na) #plots the curve

# ##option 3: find the mass for dimensions, many L's and q's ploting the curve
# Na=10 #number of points to compute
# Lmin=0.1
# Lmax=5
# NL=10
# tol=0.1

# for d in [4,5,6,7,8]:
#   for lmb in [6,8,10]:
#     amin=0.5 #minimal value of a for the plot
#     amax=1.1 #maximal value of a for the plot
#     Larray,M,Q,E=curve_mass_L(Lmin,Lmax,d,lmb,amin,amax,Na,NL) #plots the curve
#     m=-100
#     for k in range(Na):
#       for j in range (NL-1):
#         m1=M[k,j+1]-M[k,j]
#         e=max(E[k,j+1],E[k,j])
#       if m1>m:
#         if e<tol:
#           m=M[k,j+1]-M[k,j]
#     with open("derivative1.txt",'a') as arq:
#       arq.write('\n')
#       arq.write('d'+str(d)+'_lmb'+str(lmb)+'_q'+str(qmin)+'-'+str(qmax)+'_Npts'+str(Nq)+'_L'+str(Lmin)+'-'+str(Lmax)+'_NL'+str(NL))
#       arq.write('\n')
#       arq.write(str(m))
