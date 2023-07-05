import lhapdf as lha
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import os.path
import matplotlib as mpl
import math
import time
from matplotlib.backends.backend_pdf import PdfPages

###################################################################################################
###################################################################################################
                                    # BASIC FUNCTIONS
###################################################################################################
###################################################################################################

# Calls LHAPDF for the pdfset given its string formatted name, for the corresponding replica
def callpdf(pdfset, nrep):
    r = []
    for i in range(nrep):
        p = lha.mkPDF(pdfset, i)
        r.append(p)
    return r

# MC error prescription
def mc_error(xs_val):
    error = 0.5*(np.nanpercentile(xs_val[1:], 95)-np.nanpercentile(xs_val[1:],  5))
    return error

# Weighted error prescription and mean
def mc_error_rw(xs_val,weights):
    N = len(xs_val[1:])
    mean = np.sum(xs_val[1:]*weights)/N
    error = 1.642*np.std(xs_val[1:]*weights)
    return mean,error 

def weighted_mean(xs_val,weights):
    N = len(xs_val[1:])
    return 

# Generates random values normally distributed around the central value of each bin with for a given 
# stdev and returns it
def pseudodata(ipdf, iflav, stdev):
    #print('original data: ',ratios[ipdf, iflav, :nbins[iflav]-1, 0])
    pd = np.array([np.random.normal(point, stdev*point) for point in ratios[ipdf, iflav, :nbins[iflav]-1, 0]])
    #print('pseudo data: ', pd)
    return pd

# From the generated pseudodata (exp) and the replica it's compared to (theory), computes the chi
# squared function for the same experimental error considered above, along all bins of the data
def chisquared(exp, theory, error):
    chisum = 0
    N = 0
    for i,point in enumerate(theory):
        if point == 0: continue
        chisum += ((point-exp[i])/error[i])**2
        N += 1
    if N == 0: return 0
    else: return chisum/N

# Receives a PDF set, one of the two computed ratios, the generated pseudodata and the number of the
# replica it's compared to, and the error which is proportional to the unweighted central value
# and the experimental error
def chi2_rep(ipdf, iflav, irep, pd, stdev):
    cv_rep = ratios[ipdf, iflav, :nbins[iflav]-1, irep]
    cv_0 = ratios[ipdf, iflav, :nbins[iflav]-1, 0]
    return chisquared(pd, cv_rep, stdev*cv_0)

# Computation of effective number of replicas with Shannon entropy
def N_eff(ipdf, iflav, weights):
    N = pdfvec[ipdf][1]-1
    exponent = 0.
    for i in range(N):
        w = weights[ipdf][iflav][i]
        exponent += w*np.log(N/w)/N
        
    Neff = math.exp(exponent)
    return Neff

###################################################################################################
###################################################################################################
                                # PARAMETERS AND CONSTRAINTS
###################################################################################################
###################################################################################################

# Call base PDF, which is always the central NNPDF3.1 replica
basepdf = lha.getPDFSet('NNPDF31_nnlo_as_0118').mkPDF(0)


# PDF sets of the study
pdfvec = [['NNPDF31_nnlo_as_0118', 101, 'mc'],['NNPDF40_nnlo_as_01180', 101, 'mc'],['NNPDF40_nnlo_pch_as_01180', 101, 'mc'],['NNPDF31_nnlo_as_0118_1000', 1001, 'mc']]
        # Order: nnpdf31, nnpdf40, nnpdf40pch

pdf_origin = 'NNPDF31_nnlo_as_0118'

nratio = 2

npdf = len(pdfvec)

# Establish parameters of the data analysis
eps = 1e-40
errorvec = ['1','2','5','10']

# For nu_tau/nu_e @ Faser (index 1)
emin0 = 1000
emax0 = 4500
nbins0 = 8
de0 = emax0-emin0
# For nu_tau/nu_e @ SND (index 3)
emin1 = 1000
emax1 = 3500
nbins1 = 6
de1 = emax1-emin1
# Not larger energies cause stats are low

nbins = np.array([nbins0,nbins1])

elimits = np.array([[emin0,emax0],[emin1,emax1]])
de = elimits[:,1]-elimits[:,0]

###################################################################################################
###################################################################################################
                            # COMPUTATION FUNCTIONS (REWEIGHT AND ERROR)
###################################################################################################
###################################################################################################

# Given a file with neutrino data, compute the relevant numerators and denominators as a function of 
# E and \eta for the different pdf sets
def reweight_ratio():
    weights = np.zeros((npdf,nratio,2,max(nbins),nreps))
    pdfsets = []
    for ipdf in range(npdf):
        pdfsets.append(callpdf(pdfvec[ipdf][0], pdfvec[ipdf][1]))

    filepath = '/data/theorie/josepsolac/neutrino_fluxes/data_files/data_'
    hadron_dict_all = {211: 0,  113: 0,  221: 0,  1114: 0, 2224: 0, 2214: 0, 223: 0,  331: 0, 2114: 0, 
                       213: 0,  
                       323: 1,  321: 1,  311: 1,  313: 1,  333: 1,
                       411: 2,  421: 2,  431: 2,  4122: 2, 423: 2,  4224: 2, 413: 2,  433: 2,  4222: 2, 
                       4214: 2, 4212: 2, 4112: 2, 4114: 2, 441: 2,  4324: 2, 4232: 2, 4322: 2, 4314: 2, 
                       4312: 2, 4132: 2, 4332: 2, 4334: 2, 443: 2,  4422: 2, 4424: 2, 
                       3314: 3, 3324: 3, 3334: 3, 3212: 3, 3214: 3, 3122: 3, 3112: 3, 3222: 3, 3322: 3, 
                       3312: 3, 3114: 3, 3224: 3}
    
    with open(filepath + pdf_origin + '/neutrino_' + pdf_origin + '_1tev5tev_SJ.dat') as f:
        for k,line in enumerate(f):
            neutrino_ind = line.split(' ')

            # Obtains neutrino data
            stat = int(float(neutrino_ind[0]))-1
            e = float(neutrino_ind[1])
            eta = float(neutrino_ind[3])
            parent = abs(int(float(neutrino_ind[5])))
            pid = hadron_dict_all.get(parent)
            
            xs = float(neutrino_ind[2])
            ID = int(neutrino_ind[4])
            state = int(0.5-np.sign(ID)*0.5)
            flavour = int((abs(ID)-12)/2)
            id1 = int(neutrino_ind[6])
            x1 = float(neutrino_ind[7])
            id2 = int(neutrino_ind[8])
            x2 = float(neutrino_ind[9])
            Q = float(neutrino_ind[10])
            if flavour == 1: continue
            # Factor corresponding to origin pdf
            W0 = basepdf.xfxQ(id1,x1,Q)*basepdf.xfxQ(id2,x2,Q)
            nbin = np.array([math.floor(ibin) for ibin in nbins*(e-elimits[:,0])/(de)])

            for ipdf in range(npdf):
                nrep = pdfvec[ipdf][1]
                for irep in range(nrep):
                    # Weight used to reweight the cross section
                    if ipdf == 0 and irep == 0:
                        W = 1
                    else:
                        W = (pdfsets[ipdf][irep].xfxQ(id1,x1,Q)*pdfsets[ipdf][irep].xfxQ(id2,x2,Q))/W0

                    xsnew = xs*W

                    if stat == 0 and e > emin0 and e < emax0: weights[ipdf][0][flavour//2][nbin[0]][irep] += xsnew
                        
                    elif stat == 1 and e > emin1 and e < emax1: weights[ipdf][1][flavour//2][nbin[1]][irep] += xsnew 

    ratios = weights[:,:,1,:,:]/weights[:,:,0,:,:]
    # Regulator to avoid infinities in the chisquared
    ratios[np.isnan(ratios)] = 0.
    return ratios

# Given the numerators and denominators for the interesting ratios, computes the quotient and returns
# the value together with the replica errors
def errors_chdom(ratios):
    errors = np.zeros((npdf,nratio,max(nbins)))
    
    for ipdf in range(npdf):
        nrep = pdfvec[ipdf][1]
        err_prescription = pdfvec[ipdf][2]
        for iratio in range(nratio):
            for ibin in range(nbins[iratio]):
                r_val = ratios[ipdf,iratio,ibin]
                errors[ipdf,iratio,ibin] += mc_error(r_val[:nrep])           
    return errors

# Computes the weight of each replica from the chisquared obtained from the corresponding pseudodata
def reweight_pdf(ipdf, iflav, sigma):
    nrep = pdfvec[ipdf][1]
    weights_new = np.zeros(nrep-1)
    pd = pseudodata(ipdf, iflav, sigma)
    weightsum = 0

    for irep in range(1, nrep):
        chi2 = chi2_rep(ipdf, iflav, irep, pd, sigma)
        weights_new[irep-1] += (chi2**((nbins[iflav]-2)/2))*np.exp(-0.5*chi2)
        weightsum += weights_new[irep-1]
    return (weights_new)*(nrep-1)/(weightsum)

###################################################################################################
###################################################################################################
                                    # MAIN: FUNCTION CALLS
###################################################################################################
###################################################################################################

# Compute ratios from which the pseudodata will be generated and compared to
ratios = reweight_ratio()
errors = errors_chdom(ratios)

# Obtains weights from different experimental error values and writes them into weight files, along with
# the number of effective replicas
for stdstr in errorvec:
    std = float(int(stdstr))/100
    for ipdf in range(npdf):
        nrep = pdfvec[ipdf][1] 
        Neff = np.zeros((2,10))
        with open('/data/theorie/josepsolac/pythia8307/examples/weights_def/weights_'+pdfvec[ipdf][0]+'_'+stdstr+'.dat','w+') as f:
            for i in range(10):
                weights = np.zeros((npdf, nratio, nrep-1))
                for iflav in range(nratio):
                    weights[ipdf,iflav] += reweight_pdf(ipdf,iflav,std)
                    f.write('R'+'_'+str(iflav)+'_'+str(i)+' ')
                    for irep in range(nrep-1):
                        f.write(str(weights[ipdf,iflav,irep])+' ')
                    f.write('\n')
                    Neff[iflav][i] += N_eff(ipdf,iflav,weights)
        with open('/data/theorie/josepsolac/pythia8307/examples/Neff_def/Neff_'+pdfvec[ipdf][0]+'_'+stdstr+'.dat','w+') as f:
            for i in range(10):
                f.write('Iteration ' + str(i) + '; Ratio 1: ' + str(Neff[0][i]) + '; Ratio 2: ' + str(Neff[1][i])+'\n')