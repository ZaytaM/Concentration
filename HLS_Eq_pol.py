# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:16:38 2020

@author: Mathieu Lewin
"""


from numpy import *
from numpy.linalg import norm, solve
#import matplotlib 
#matplotlib.use('Agg')
from matplotlib.pyplot import plot, figure, title, savefig, subplots
from scipy.optimize import minimize
from scipy.special import gamma,hyp2f1
from scipy.integrate import quad
from pickle import dump, load
import os.path
from multiprocessing import Pool,cpu_count
import time
import datetime

# Optimize the best constant in the reversed Hardy-Littlewood-Sobolev inequality of
# Carillo, Delgadino, Dolbeault, Frank & Hoffmann, J. Math. Pures Appl. (2019)
# as explained in the forthcoming article by
# Carillo, Delgadino, Frank & Lewin

#Here we try to find a solution of the Euler-Lagrange equation behaving like $r^{-2/(1-q)}$ at the origin and then we compute its mass
#If the mass is <1 we get a solution of the original problem with concentration
#Rho is taken of the special form Rho=r^{-2/(1-q)}(1+r)^{(2-lmb)/(1-q)}f(r/(1+r))^{-1/(1-q)} 
#with f(y)=pt+P(y)^2+Q(y)^2 where P and Q are polynomials and pt is a small number

##############################################################
##############################################################
####PARAMETERS TO BE DEFINED FOR THE COMPUTATION
Nc=10 #degree of P and Q in approx of rho (see above). For lmb=2k, need Nc>=k
Rmax=20 # computation of the integrals on [0,Rmax]
N=1000 #Nb of discretization points for the integrals
##parameters for the optimization
use_exact_gradient=True #uses exact gradient in minimization
tol_BFGS=1e-10 #error to be used in BFGS
pt=1e-8 #small number used to make sure that f>0, just in case
N_rand=10 #number of random initial conditions to be used in minimization, in addition to C_init
be_silent=False #writes less messages when True
##parameters for disk usage
save_fig=True #whether the figures and results are saved on the disk or not
##############################################################
##############################################################


##############################################################
##############################################################
####DISCRETIZATION GRID AS A GLOBAL VARIABLE
global R
global DR
R_linear=arange(1/N,1+1/N,1/N)*Rmax #grid
R=R_linear
DR=Rmax/N
Y=R/(1+R)
YY=[Y**i for i in range(Nc)]
##############################################################
##############################################################


##############################################################
##############################################################
####FUNCTIONS
##############################################################
##############################################################

#computes the area of the hypersphere S^(d-1) in R^d
def sphere(d):
    return 2*pi**(d/2)/gamma(d/2)

#computes q in terms of a
def q(d,a,lmb):
    return d*(2-a)/(d*(2-a)+lmb)

#critical alpha below which it is known that there cannot be condensation
def a_crit(d,lmb):
    return max(0,(4*d - (d - 2)*lmb)/2/d)

#computes the matrix K_ij}=|S^{d-1}|^{-2}\iint|R_i\omega-R_j\omega'|^lambda d\omega d\omega'
def Kd(d,lmb):
    if lmb==2:
        K=outer(R**2,ones(N))+outer(ones(N),R**2)
    elif lmb==4:
        K=outer(R**4,ones(N))+outer(ones(N),R**4)+(2+4/d)*outer(R**2,R**2)
    elif lmb==6:
        K=outer(R**6,ones(N))+outer(ones(N),R**6)+(3+12/d)*(outer(R**4,R**2)+outer(R**2,R**4))
    elif lmb==8:
        K=outer(R**8,ones(N))+outer(ones(N),R**8)+(32+48/d+4*d)/(2+d)*(outer(R**6,R**2)+outer(R**2,R**6))+(60+144/d+6*d)/(2+d)*outer(R**4,R**4)
    elif lmb==10:
        K=outer(R**(10),ones(N))+outer(ones(N),R**(10))+(50+80/d+5*d)/(2+d)*(outer(R**8,R**2)+outer(R**2,R**8))+(140+480/d+10*d)/(2+d)*(outer(R**6,R**4)+outer(R**4,R**6))
    elif d==3:
        I=array([R,]*N)
        J=transpose(I)
        IJinv=outer(R**(-1),R**(-1))
        K=multiply((I+J)**(lmb+2)-abs(I-J)**(lmb+2),IJinv)/2/(lmb+2)
    else:
        Kfun=lambda i, j: (R[i]**2+R[j]**2)**(lmb/2)*hyp2f1((2-lmb)/4,-lmb/4,d/2,4*R[i]**2*R[j]**2/(R[i]**2+R[j]**2)**2)
        K=fromfunction(Kfun, (N, N), dtype=int)
    return K

#matrix used to compute \rho\ast|x|^\lambda / r^2/(1+r)^(lmb-2)
def KCd(d,a,lmb,K):
    qq=q(d,a,lmb)
    Cfun=lambda i, j: (K[i,j]-R[j]**lmb)/R[i]**2/(1+R[i])**(lmb-2)*R[j]**(d-1-2/(1-qq))*(1+R[j])**((2-lmb)/(1-qq))
    KC=fromfunction(Cfun, (N, N), dtype=int)
    return KC

# Rho^(q-1)=r^2(1+r)^(lmb-2)f(r/(1+r)) where f(y)=pt+P(y)**2+Q(y)**2
def fp(C):
    return pt+dot(C[:Nc],YY)**2+dot(C[Nc:],YY)**2        

# Rho^(q-1)=r^2(1+r)^(lmb-2)f(r/(1+r)) where f(y)=pt+P(y)**2+Q(y)**2
def Rho(C,d,a,lmb):
    qq=q(d,a,lmb)
    return R**(-2/(1-qq))*(1+R)**((2-lmb)/(1-qq))*fp(C)**(-1/(1-qq))

#we try to solve f=Phi(f)
def Phi(C,d,a,lmb,KC):
    qq=q(d,a,lmb)
    f_inv=fp(C)**(-1/(1-qq))
    Phi1=sphere(d)*dot(KC,f_inv)*DR + (1 -sphere(d)*sum(R**(d-1-2/(1-qq))*(1+R)**((2-lmb)/(1-qq))*f_inv)*DR)*R**(lmb-2)*(1+R)**(2-lmb)
    return (1-qq)/qq*maximum(Phi1,pt)

#simple discretized mass of Rho
def mass(C,d,a,lmb):
    qq=q(d,a,lmb)
    f_inv=fp(C)**(-1/(1-qq))
    return sphere(d)*sum(R**(d-1-2/(1-qq))*(1+R)**((2-lmb)/(1-qq))*f_inv*DR)

#mass of Rho, with the integral computed to a high precision, takes more time
def mass2(C,d,a,lmb):
    qq=q(d,a,lmb)
    ff=lambda r: pt+sum([C[j]*(r/(1+r))**j for j in range(Nc)])**2+sum([C[Nc+j]*(r/(1+r))**j for j in range(Nc)])**2
    fun=lambda r: sphere(d)*r**(d-1-2/(1-qq))*(1+r)**((2-lmb)/(1-qq))*ff(r)**(-1/(1-qq))
    return quad(fun,0,+inf)[0]

#we minimize this function
def func_min(C,d,a,lmb,KC):
    qq=q(d,a,lmb)
    f_inv=fp(C)**(-1/(1-qq))
    Phi_inv=Phi(C,d,a,lmb,KC)**(-1/(1-qq))
    return sum(R**(d-1-2/(1-qq))*(1+R)**((2-lmb)/(1-qq))*(f_inv-Phi_inv)**2*DR)

#same with gradient
def func_min_grad(C,d,a,lmb,KC):
    qq=q(d,a,lmb)
    gf=zeros(2*Nc)
    pol1=dot(C[:Nc],YY)
    pol2=dot(C[Nc:],YY)
    f=pt+pol1**2+pol2**2
    f_inv=f**(-1/(1-qq))
    Ph=sphere(d)*dot(KC,f_inv*DR) + (1 -sphere(d)*sum(R**(d-1-2/(1-qq))*(1+R)**((2-lmb)/(1-qq))*f_inv*DR))*R**(lmb-2)*(1+R)**(2-lmb)
    Ph=(1-qq)/qq*maximum(Ph,pt)
    Phi_inv=Ph**(-1/(1-qq))
    fn=sum(R**(d-1-2/(1-qq))*(1+R)**((2-lmb)/(1-qq))*(f_inv-Phi_inv)**2*DR)
    for j in range(Nc):
        #derivative with respect to C_j
        dCj_f=2*pol1*Y**j
        dCj_f_inv=-f**(-1/(1-qq)-1)/(1-qq)*dCj_f
        dCj_Phi=(1-qq)/qq*(sphere(d)*dot(KC,dCj_f_inv*DR)  -sphere(d)*sum(R**(d-1-2/(1-qq))*(1+R)**((2-lmb)/(1-qq))*dCj_f_inv*DR)*R**(lmb-2)*(1+R)**(2-lmb))
        dCj_Phi_inv=-Ph**(-1/(1-qq)-1)/(1-qq)*dCj_Phi
        gf[j]=2*sum(R**(d-1-2/(1-qq))*(1+R)**((2-lmb)/(1-qq))*(f_inv-Phi_inv)*(dCj_f_inv-dCj_Phi_inv)*DR)
        #derivative with respect to C_{Nc+j}
        dCj_f=2*pol2*Y**j
        dCj_f_inv=-f**(-1/(1-qq)-1)/(1-qq)*dCj_f
        dCj_Phi=(1-qq)/qq*(sphere(d)*dot(KC,dCj_f_inv*DR)  -sphere(d)*sum(R**(d-1-2/(1-qq))*(1+R)**((2-lmb)/(1-qq))*dCj_f_inv*DR)*R**(lmb-2)*(1+R)**(2-lmb))
        dCj_Phi_inv=-Ph**(-1/(1-qq)-1)/(1-qq)*dCj_Phi
        gf[Nc+j]=2*sum(R**(d-1-2/(1-qq))*(1+R)**((2-lmb)/(1-qq))*(f_inv-Phi_inv)*(dCj_f_inv-dCj_Phi_inv)*DR)
    return fn, gf

#minimizing func_min
def find_opt(C_init,d,a,lmb,KC):
    if use_exact_gradient==True:
        res=minimize(func_min_grad,C_init,args=(d,a,lmb,KC),method='BFGS',jac=True,tol=tol_BFGS)
    else:
        res=minimize(func_min,C_init,args=(d,a,lmb,KC),method='BFGS',tol=tol_BFGS)
    return res.x, res.fun

#finds the solution and returns its mass, the L^1 error in equation and the optimal C
def find_mass(d,a,lmb,print_mess=True,C_init=random.rand(2*Nc)):
    K=Kd(d,lmb)
    KC=KCd(d,a,lmb,K)
    t1=time.time()
    if print_mess==True:
        print('Computing the mass for d=',d,'lmb=',lmb,'a=',a)
        print('Degree of polynomials=',Nc)
        print('Nb of disc. points for integrals=',N,'Rmax=',Rmax)
        print('Using', N_rand,'random initial conditions')
    qq=q(d,a,lmb)
    value_fn=float("inf")
    #minimize for C_init and N_rand initial conditions. Need more random conditions when lambda gets large
    for _ in range(N_rand+1):
        C,value2=find_opt(C_init,d,a,lmb,KC)
        if value2<value_fn:
            Copt=C
            value_fn=value2
        C_init=random.rand(2*Nc)
    f_inv=fp(Copt)**(-1/(1-qq))
    Phi_inv=Phi(Copt,d,a,lmb,KC)**(-1/(1-qq))
    mass=sphere(d)*sum(R**(d-1-2/(1-qq))*(1+R)**((2-lmb)/(1-qq))*f_inv*DR)    
    diff_mass=sphere(d)*sum(R**(d-1-2/(1-qq))*(1+R)**((2-lmb)/(1-qq))*abs(f_inv-Phi_inv)*DR)
    if print_mess==True:
        print('Found mass=',mass,'with error=',diff_mass)
        print('Computation in...',int(100*(time.time()-t1))/100,'sec')        
    return mass, diff_mass, Copt
        
#main function to plot the mass of the fixed point solution on [amin,amax] with Na points
def curve_mass(d,lmb,amin,amax,Na):
    t1=time.time()
    print('Computing the mass of the singular solution in',d,'d at lmb=',lmb,'...')
    print('Degree of polynomials=',Nc)
    print('Nb of disc points for integrals=',N,'Rmax=',Rmax)
    print('Using', N_rand,'random initial conditions')
    A=arange(amin,amax,(amax-amin)/Na)
    M=zeros(len(A))
    diffM=zeros(len(A))
    Q=zeros(len(A))
    for i in range(len(A)):
        #computation of the matrix C...
        Q[i]=q(d,A[i],lmb)        
        mopt,diffmopt,Copt=find_mass(d,A[i],lmb,print_mess=False)
        M[i]=mopt
        diffM[i]=diffmopt
        if be_silent==False:
            print('At a=',A[i], 'found mass=',mopt,'with error=',diffmopt)
    #plots the mass in terms of a and q
    fig1, ax1=subplots(1)
    ax1.plot(A,M,'b')
    ax1.plot(A,ones(len(A)),'k')
    fig2, ax2=subplots(1)
    ax2.plot(Q,M,'b')
    ax2.plot(Q,ones(len(A)),'k')
    #saves everything on the disk
    if save_fig==True: 
        basefilename='d'+str(d)+'_lmb'+str(lmb)+'_a'+str(amin)+'-'+str(amax)+'_Npts'+str(N)+'_deg'+str(Nc)
        fig1.savefig(basefilename+'_AM'+'.pdf')
        fig2.savefig(basefilename+'_QM'+'.pdf')
        dump([A,M,diffM,Q], open(basefilename+'.p', "wb"))        
    print('Computation in...',int(100*(time.time()-t1))/100,'sec')
    return A,M,diffM,Q

#print some results in the file flnm            
def myprint(flnm,s):
    file = open(flnm,"a+")
    file.write("\n" + s)
    file.close()
    
#function to optimize on one processor
def optimize_cpu(my_input):
    random.seed()
    i,d,a,lmb=my_input
    t1=time.time()
    mopt,diffmopt,Copt=find_mass(d,a,lmb,print_mess=False)
    if be_silent==False:
        print('At a=',a,'found mass=',mopt,'with error=',diffmopt)
    tobeprinted="a="+str(a)+" mass="+str(mopt)+" error="+str(diffmopt)
    txtfilename='cpu_d'+str(d)+'_lmb'+str(lmb)+'_a'+str(amin)+'-'+str(amax)+'_N'+str(N)+'_deg'+str(Nc)+'_Nrun'+str(Nrun)+'.txt'
    myprint(txtfilename,tobeprinted)
    return i, mopt, diffmopt, Copt

#same as curve_mass but uses parallel computing, each processor computes Nrun points
def curve_mass_cpu(d,lmb,amin,amax,Nrun):
    t1=time.time()
    Ncpu=cpu_count()
    Na=Nrun*Ncpu
    print('Computing the mass of the singular solution in',d,'d at lmb=',lmb,'...')
    print('Degree of polynomials=',Nc)
    print('Nb of disc points for integrals=',N,'Rmax=',Rmax)
    print('Using', N_rand,'random initial conditions')
    print('Computation of',str(Ncpu*Nrun),'points in parallel on',Ncpu,'cpus')
    A=amin+arange(Na)/(Na-1)*(amax-amin)
    M=zeros(Na)
    diffM=zeros(Na)
    Q=array([q(d,A[i],lmb) for i in range(Na)])
    for k in range(Nrun):
        #sends the computations to the Ncpu processors
        with Pool(Ncpu) as pool:
            list_res=pool.map( optimize_cpu, [(Ncpu*k+i,d,A[Ncpu*k+i],lmb) for i in range(Ncpu)])
            #collects the results and saves them in the vectors M and diffM
            for LL in list_res:
                M[LL[0]]=LL[1]
                diffM[LL[0]]=LL[2]
    #plots the mass in terms of a and q
    fig1, ax1=subplots(1)
    ax1.plot(A,M,'b')
    ax1.plot(A,ones(Na),'k')
    fig2, ax2=subplots(1)
    ax2.plot(Q,M,'b')
    ax2.plot(Q,ones(Na),'k')
    #saves everything on the disk
    if save_fig==True: 
        basefilename='cpu_d'+str(d)+'_lmb'+str(lmb)+'_a'+str(amin)+'-'+str(amax)+'_N'+str(N)+'_deg'+str(Nc)+'_Nrun'+str(Nrun)
        fig1.savefig(basefilename+'_AM'+'.pdf')
        fig2.savefig(basefilename+'_QM'+'.pdf')
        dump([A,M,diffM,Q], open(basefilename+'.p', "wb"))        
    print('Computation in...',int(100*(time.time()-t1))/100,'sec')
    return A,M,diffM,Q

#function to find the a such that the mass equals 1, with given error
#[amin amax] is the initial interval where to look for a
#The interval will be changed if bad, but it is always better to start with a good guess
def crit_a(d,lmb,amin,amax,error,print_mess=True):
    t1=time.time()
    if print_mess==True:
        print('Computing the critical a in',d,'d at lmb=',lmb,'...')
        print('Degree of polynomials=',Nc)
        print('Nb of disc points for integrals=',N,'Rmax=',Rmax)
        print('Using', N_rand,'random initial conditions')
        print('Initial a=',amin,amax)
    #computes initial masses
    m_min,mdiff_min,C_min=find_mass(d,amin,lmb,print_mess=False)
    m_max,mdiff_max,C_max=find_mass(d,amax,lmb,print_mess=False)
    if print_mess==True:
        print('Initial masses=',m_min,m_max)
    #check that the initial points are good and otherwise change them
    if m_min<1:
        if print_mess==True:
            print('Try changing left point...')
        amax,m_max,mdiff_max,C_max=amin,m_min,mdiff_min,C_min
        tot=0
        while m_min<1 and tot<10:
            amin=max(amin/2,0.1)
            m_min,mdiff_min,C_min=find_mass(d,amin,lmb,print_mess=False)
            tot=tot+1
    if m_max>1:
        if print_mess==True:
            print('Try changing right point...')
        amin,m_min,mdiff_min,C_min=amax,m_max,mdiff_max,C_max
        tot=0
        while m_max>1 and tot<10:
            amax=min(amax*2,1.5)
            m_max,mdiff_max,C_max=find_mass(d,amax,lmb,print_mess=False)    
            tot=tot+1
    #starts bisection if points are good
    if m_min>1 and m_max<1:
        tot=0
        while abs(m_max-m_min)>error and tot<10:
            anew=(amin+amax)/2
            m_new,mdiff_new,C_new=find_mass(d,anew,lmb,print_mess=False)
            if m_new>1:
                amin,m_min,mdiff_min,C_min=anew,m_new,mdiff_new,C_new
            else:
                amax,m_max,mdiff_max,C_max=anew,m_new,mdiff_new,C_new
            if print_mess==True:
                print('For a=',amin,amax)
                print('Masses=',m_min,m_max)
            tot=tot+1
        a=(amin+amax)/2
        m,mdiff,C=find_mass(d,a,lmb,print_mess=False,C_init=C_min)
    else:#computation failed
        a=(amin+amax)/2
        m=(m_min+m_max)/2
        mdiff=float('inf')
        C=(C_min+C_max)/2
    if print_mess==True:
        print('At a=',a,'mass=',m,'with error=',mdiff,'q=',q(d,a,lmb))
        print('Exact mass of this density=',mass2(C_new,d,anew,lmb))
        print('Computation in...',int(100*(time.time()-t1))/100,'sec')
    return a,m,mdiff,C


#############################################################################
##END OF FUNCTIONS
#############################################################################

######################################################
###OPTION 1: calculation of one mass
######################################################
#d=5
#lmb=5
#a=1
#find_mass(d,a,lmb)

######################################################
###OPTION 2: calculation of the mass curve in terms of a/q (d, lmb fixed)
######################################################
#d=5
#lmb=5
#amin=0.5 #minimal value of a for the plot
#amax=1.1 #maximal value of a for the plot
#Na=8 #total number of points to compute
#curve_mass(d,lmb,amin,amax,Na) #plots the curve

######################################################
###OPTION 3: same but uses parallel computing
######################################################
#d=5
#lmb=5
#amin=0.5 #minimal value of a for the plot
#amax=1.1 #maximal value of a for the plot
#Nrun=1 #number of points to compute per processor
#curve_mass_cpu(d,lmb,amin,amax,Nrun) #plots the curve

######################################################
###OPTION 4: calculation of the critical a at given d, lmb
######################################################
d=5
lmb=5
amin=0.87 #starting lower bound on a
amax=0.88 #starting upper bound on a
error_mass=0.01 #allowed error for the mass
crit_a(d,lmb,amin,amax,error_mass)
