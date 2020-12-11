# Concentration

This repository contains the python code related to the manuscript "Fast Diffusion leads to partial mass concentration in Keller-Segel type models" by J. A. Carrillo, M. G. Delgadino, R. L. Frank, and M. Lewin.

In the Results folder you can find the output of the scripts.

The script even_lmb.py deals with the cases of lambda=4, 6, 8 and 10:

-) Option 1 finds the mass for just one specific value of alpha.

-) Option 2 calculates the mass of the solution to the Euler-Lagrange equation with parameter L=0 for different values of alpha. The output can be found in the folder mass within the Results. Some of these images are included in figure 1 of the paper.

-) Option 3 calculates the mass of the solution to the Euler-Lagrange equation for different values of L and q. It checks the monotonicity of the mass with respect to parameter L for a fixed value of q. The output can be found in the folder monotonicity within the Results.


The script HLS_Eq_Pol.py deals with any lambda for L=0, with the density approximated by a polynomial as explained in the paper. For simplicity the integrals are now discretized on a regular grid.

-) Option 1 finds an approximate solution to the Euler-Lagrange equation with L=0 and computes its mass, for just one specific value of alpha.

-) Option 2 does the same for different values of alpha.

-) Option 3 does the same in parallel on several CPUs

-) Option 4 computes the critical alpha for a given lambda, using a simple bisection method. The outputs in dimension 5 for 1000 discretization points can be read in the file crit_a_d5_N1000.txt in the folder Critical_q within the Results. These are the data used to plot the curve in Figure 2 of the paper.
