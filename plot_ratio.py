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

# Different error prescriptions 
def symhess_error(xs_val):
    # Calculates error for one bin
    error = math.sqrt(np.variance(xs_val[1:]))
    return 1.642*error

def hess_error(xs_val):
    # Calculates error for one bin
    error = np.sqrt(np.sum(np.square(xs_val[2::2] - xs_val[1::2])))/2
    return error

def mc_error(xs_val):
    error = 0.5*(np.nanpercentile(xs_val[1:], 95)-np.nanpercentile(xs_val[1:],  5))
    return error

###################################################################################################
###################################################################################################
                                # PARAMETERS AND CONSTRAINTS
###################################################################################################
###################################################################################################

# Call base PDF, which is always the central NNPDF3.1 replica
basepdf = lha.getPDFSet('NNPDF31_nnlo_as_0118').mkPDF(0)

# PDF sets of the study
pdfvec = [['NNPDF31_nnlo_as_0118', 101, 'mc'],['CT18NNLO', 59, 'hess'],['MSHT20nnlo_as118', 65, 'hess'],['NNPDF40_nnlo_as_01180', 101, 'mc'],['NNPDF40_nnlo_pch_as_01180', 101, 'mc']]
          #Order: nnpdf31, ct18, msht, nnpdf40, nnpdf40pch

# Establish parameters of the data analysis
pdf_origin = pdfvec[0][0]
nratio = 2

npdf = len(pdfvec)

# Energy limits of charm domination
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
    # Empty initial array to store cross section weighted events, and the number of events in each bin,
    # to ensure sufficient statistics
    weights = np.zeros((npdf,nratio,2,max(nbins),101))
    nevents = np.zeros((nratio,2,max(nbins)))
    pdfsets = []

    # Call all PDF sets and replicas
    for ipdf in range(npdf):
        pdfsets.append(callpdf(pdfvec[ipdf][0], pdfvec[ipdf][1]))

    # Particle library appearing in the particle record and filepath of neutrino data
    filepath = '/data/theorie/josepsolac/neutrino_fluxes/data_files/data_'
    hadron_dict_all = {211: 0,  113: 0,  221: 0,  1114: 0, 2224: 0, 2214: 0, 223: 0,  331: 0, 2114: 0, 
                       213: 0,  
                       323: 1,  321: 1,  311: 1,  313: 1,  333: 1,
                       411: 2,  421: 2,  431: 2,  4122: 2, 423: 2,  4224: 2, 413: 2,  433: 2,  4222: 2, 
                       4214: 2, 4212: 2, 4112: 2, 4114: 2, 441: 2,  4324: 2, 4232: 2, 4322: 2, 4314: 2, 
                       4312: 2, 4132: 2, 4332: 2, 4334: 2, 443: 2,  4422: 2, 4424: 2, 
                       3314: 3, 3324: 3, 3334: 3, 3212: 3, 3214: 3, 3122: 3, 3112: 3, 3222: 3, 3322: 3, 
                       3312: 3, 3114: 3, 3224: 3}
    
    # Run through data file containing neutrinos between 1 and 5 TeV of energy, to reduce computational cost
    with open(filepath + pdf_origin + '/neutrino_' + pdf_origin + '_1tev5tev_SJ.dat') as f:
        for k,line in enumerate(f):
            neutrino_ind = line.split(' ')

            # Obtains neutrino data from file
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

            # Discard muon neutrinos since they don't partake in the study
            if flavour == 1: continue

            # Factor corresponding to origin pdf
            W0 = basepdf.xfxQ(id1,x1,Q)*basepdf.xfxQ(id2,x2,Q)

            # Number of bin as obtained from the energy
            nbin = np.array([math.floor(ibin) for ibin in nbins*(e-elimits[:,0])/(de)])

            for ipdf in range(npdf):
                nrep = pdfvec[ipdf][1]
                for irep in range(nrep):
                    # Computation of the weight used to reweight the cross section
                    if ipdf == 0 and irep == 0:
                        W = 1
                    else:
                        W = (pdfsets[ipdf][irep].xfxQ(id1,x1,Q)*pdfsets[ipdf][irep].xfxQ(id2,x2,Q))/W0
                    
                    # Cross section reweighting
                    xsnew = xs*W

                    # Assign the corresponding neutrino component to either the denominator (in the case
                    # of e neutrinos) or the numerator (in the case of tau neutrinos), according to the 
                    # pertinent detector and energy range
                    if stat == 0 and e > emin0 and e < emax0: 
                        weights[ipdf][0][flavour//2][nbin[0]][irep] += xsnew
                        if ipdf == 0 and irep == 0: nevents[0][flavour//2][nbin[0]] += 1
                    elif stat == 1 and e > emin1 and e < emax1: 
                        weights[ipdf][1][flavour//2][nbin[1]][irep] += xsnew 
                        if ipdf == 0 and irep == 0: nevents[1][flavour//2][nbin[1]] += 1

    # Finally computes ratios from the ratio components dividing tau by electron
    ratios = weights[:,:,1,:,:]/weights[:,:,0,:,:]
    return ratios, nevents

# Given the numerators and denominators for the interesting ratios, computes the quotient and returns
# the value together with the replica errors, with the corresponding error prescription
def errors_chdom(ratios):
    errors = np.zeros((npdf,nratio,max(nbins)))
    
    for ipdf in range(npdf):
        nrep = pdfvec[ipdf][1]
        err_prescription = pdfvec[ipdf][2]
        for iratio in range(nratio):
            for ibin in range(nbins[iratio]):
                r_val = ratios[ipdf,iratio,ibin]
                if err_prescription == 'mc':
                    errors[ipdf,iratio,ibin] += mc_error(r_val[:nrep])
                elif err_prescription == 'hess':
                    errors[ipdf,iratio,ibin] += hess_error(r_val[:nrep])
                elif err_prescription == 'symhess':
                    errors[ipdf,iratio,ibin] += symhess_error(r_val[:nrep])             
    return errors

###################################################################################################
###################################################################################################
                                    # PLOTTING FUNCTIONS
###################################################################################################
###################################################################################################

# Plot the neutrino flavour ratios for all PDF sets as a function of energy with the corresponding
# PDF errors
def mkplot_ratio_flavour(elimits,ratios,errors,histname):
    pp = PdfPages(histname+'.pdf')
    ncols = 2
    nrows = nratio
    eps = 0.3   
    fig, axs = plt.subplots(nrows, ncols, sharex=False, figsize=(ncols*5,nrows*3.5))

    r = ratios[:, :, :, 0]
    e = errors[:, :, :]

    tag = r'$\nu_{\tau}/\nu_e$'
    pdfset = ['NNPDF3.1 (ref)','CT18','MSHT','NNPDF4.0','NNPDF4.0 P.Ch.']
    detectors = ['FASER'+r'$\nu$', 'SND']
    linestyle_tuple = [('dotted', (0, (1, 1))), ('dashed', (0, (5, 5))), ('dashdotted',(0, (4, 3, 1, 3))),
                       ('dashdotdotted', (0, (3, 3, 1, 3, 1, 3))), ('solid', 'solid')]
    rescolors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    
   # Upper plot: original vs reweighted cross sections as a function of energy
    fig.suptitle('Flavour ratios @ charm domination (normalized to NNPDF3.1)', fontweight='bold', fontsize=14, y=1.06)

    p = []

    for plot, ax in enumerate(axs.flatten()):
        M = int(2*(plot%2))
        iflav = int(plot//2)
        emin, emax = elimits[iflav]
        energies = np.linspace(emin,emax,nbins[iflav])
        ls_name, ls = linestyle_tuple[0]
        refval = r[0][iflav]
        
        if plot == 0: 
            p1 = ax.plot(np.NaN, np.NaN, ls = ls, color = rescolors[0], alpha = 1)
            p2 = ax.fill(np.NaN, np.NaN, color = rescolors[0], alpha=0.2)
            p.append(p1)
            p.append(p2)
        ax.hist(energies[:-1], energies, weights = np.divide(r[0][iflav][:nbins[iflav]], refval[:nbins[iflav]])[:-1], histtype = 'step', color = rescolors[0], ls = ls, alpha = 1)
        ax.bar(energies[:-1], height = 2*np.divide(e[0][iflav][:nbins[iflav]],refval[:nbins[iflav]])[:-1], width = energies[1:]-energies[:-1], 
                bottom = np.divide(r[0][iflav][:nbins[iflav]]-e[0][iflav][:nbins[iflav]],refval[:nbins[iflav]])[:-1], 
                align = 'edge', color = rescolors[0], alpha = 0.2)
        for ipdf in range(M+1,M+3):
            ls_name, ls = linestyle_tuple[ipdf]
            p1 = ax.plot(np.NaN, np.NaN, ls = ls, color = rescolors[ipdf], alpha = 1)
            p2 = ax.fill(np.NaN, np.NaN, color = rescolors[ipdf], alpha=0.2)
            ax.hist(energies[:-1], energies, weights = np.divide(r[ipdf][iflav][:nbins[iflav]], refval[:nbins[iflav]])[:-1], histtype = 'step', color = rescolors[ipdf], ls = ls, alpha = 1)
            ax.bar(energies[:-1], height = 2*np.divide(e[ipdf][iflav][:nbins[iflav]],refval[:nbins[iflav]])[:-1], width = energies[1:]-energies[:-1], 
                   bottom = np.divide(r[ipdf][iflav][:nbins[iflav]]-e[ipdf][iflav][:nbins[iflav]],refval[:nbins[iflav]])[:-1], 
                   align = 'edge', color = rescolors[ipdf], alpha = 0.2)
            p.append(p1)
            p.append(p2)

        #ax.set_xscale('log')
        
        ax.set_xlabel('E (TeV)', fontsize=12)
        ax.set_ylim(1-eps,1+eps)
        ax.set_ylabel(tag, fontsize=12)

        textstr = tag + ' at ' + detectors[iflav]
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(0.05, 0.9, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)

    axs[0, 0].set_xlim(emin0, emax0)
    axs[0, 0].set_xticks([1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500])
    axs[0, 0].set_xticklabels([r'1.0', r'1.5', r'2.0', r'2.5', r'3.0', r'3.5', r'4.0', r'4.5'])
    axs[0, 0].minorticks_off()

    axs[0, 1].set_xlim(emin0, emax0)
    axs[0, 1].set_xticks([1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500])
    axs[0, 1].set_xticklabels([r'1.0', r'1.5', r'2.0', r'2.5', r'3.0', r'3.5', r'4.0', r'4.5'])
    axs[0, 1].minorticks_off()

    axs[1, 0].set_xlim(emin1, emax1)
    axs[1, 0].set_xticks([1000, 1500, 2000, 2500, 3000, 3500])
    axs[1, 0].set_xticklabels([r'1.0', r'1.5', r'2.0', r'2.5', r'3.0', r'3.5'])
    axs[1, 0].minorticks_off()

    axs[1, 1].set_xlim(emin1, emax1)
    axs[1, 1].set_xticks([1000, 1500, 2000, 2500, 3000, 3500])
    axs[1, 1].set_xticklabels([r'1.0', r'1.5', r'2.0', r'2.5', r'3.0', r'3.5'])
    axs[1, 1].minorticks_off()

    # axs[0,0].set_title('Relative flavour yield for Hessian sets')
    # axs[0,1].set_title('Relative flavour yield for Monte Carlo sets')

    fig.legend([(p[0][0],p[1][0]),(p[2][0],p[3][0]),(p[4][0],p[5][0]),(p[6][0],p[7][0]),(p[8][0],p[9][0])],
               [pdfset[0],pdfset[1],pdfset[2],pdfset[3],pdfset[4]],
                frameon=True, ncol=5, bbox_to_anchor=(0.5,1.015), loc=9, prop={'size':14})

    fig.tight_layout()
    fig.show()
    pp.savefig(fig, bbox_inches='tight')
    pp.close()

    return

###################################################################################################
###################################################################################################
                                    # MAIN: FUNCTION CALLS
###################################################################################################
###################################################################################################

fraction = ['denominator','numerator']

ratios, nevents = reweight_ratio()
errors = errors_chdom(ratios)

mkplot_ratio_flavour(elimits,ratios,errors,'ratios_tau2e')

with open('/data/theorie/josepsolac/pythia8307/examples/nevents.dat','w+') as f:
    for i in range(nratio):
        for j in range(2):
            f.write('ratio '+str(i)+' '+fraction[j]+'\n')
            for k in range(nbins[i]-1):
                f.write(str(int(nevents[i,j,k]))+'\n')