import lhapdf as lha
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import os.path
import matplotlib as mpl
import time
from matplotlib.backends.backend_pdf import PdfPages
import cProfile

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
    return error

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
pdfvec = [['NNPDF31_nnlo_as_0118', 101, 'mc'],['CT18NNLO', 59, 'hess'],['MSHT20nnlo_as118', 65, 'hess'], ['NNPDF40_nnlo_as_01180', 101, 'mc'],['NNPDF40_nnlo_pch_as_01180', 101, 'mc']]

pdf_origin = 'NNPDF31_nnlo_as_0118'
npdf = len(pdfvec)

# Establish parameters of the data analysis
njobs = 15000
nbins = 26

# For the benchmark neutrino fluxes, the full energy spectrum is studied
emin = 10
emax = 5000

# Integrated luminosity for LHC Run III
L = 1.5*(10**14)

###################################################################################################
###################################################################################################
                            # COMPUTATION FUNCTIONS (REWEIGHT AND ERROR)
###################################################################################################
###################################################################################################

# Given a file with neutrino data, compute the cross section histograms as a function of E and \eta
# and reweight them into all the PDF sets contained in pdfvec
def reweight_flux():
    pdfsets = []
    npdf = len(pdfvec)

    # Call all PDF sets and replicas
    for i in range(npdf):
        pdfsets.append(callpdf(pdfvec[i][0], pdfvec[i][1]))

    # Empty initial array to store cross section weighted events
    weights = np.zeros((npdf,2,3,2,4,nbins,101))

    # Particle library appearing in the particle record and filepath of neutrino data
    filepath = '/data/theorie/josepsolac/neutrino_fluxes/data_files/data_'
    hadron_dict_all = {211: 0,  113: 0,  221: 0,  1114: 0, 2224: 0, 2214: 0, 223: 0,  331: 0, 2114: 0, 
                       213: 0,  
                       323: 1,  321: 1,  311: 1,  313: 1,  333: 0, 
                       411: 2,  421: 2,  431: 2,  4122: 2, 423: 2,  4224: 2, 413: 2,  433: 2,  4222: 2, 
                       4214: 2, 4212: 2, 4112: 2, 4114: 2, 441: 2,  4324: 2, 4232: 2, 4322: 2, 4314: 2, 
                       4312: 2, 4132: 2, 4332: 2, 4334: 2, 443: 2,  4422: 2, 4424: 2, 
                       3314: 3, 3324: 3, 3334: 3, 3212: 3, 3214: 3, 3122: 3, 3112: 3, 3222: 3, 3322: 3, 
                       3312: 3, 3114: 3, 3224: 3}

    k=0
    # Run through 15K jobs, enough neutrinos to make statistically accurate fluxes
    with open(filepath + pdf_origin + '/neutrino_' + pdf_origin + '_full_15K.dat') as f:
        for k,line in enumerate(f):
            neutrino_ind = line.split(' ')

            # Obtains neutrino data from file
            stat = int(neutrino_ind[0])-1
            e = float(neutrino_ind[1])

            # Only keep neutrinos within energy range and particle library (excluding bottom particles)
            if e>emax: continue
            eta = float(neutrino_ind[3])
            parent = abs(int(neutrino_ind[5]))
            pid = hadron_dict_all.get(parent, 99)
            if pid > 4: continue
            xs = float(neutrino_ind[2])
            ID = int(neutrino_ind[4])
            state = int(0.5-np.sign(ID)*0.5)
            flavour = int((abs(ID)-12)/2)
            id1 = int(neutrino_ind[6])
            x1 = float(neutrino_ind[7])
            id2 = int(neutrino_ind[8])
            x2 = float(neutrino_ind[9])
            Q = float(neutrino_ind[10])

            # Factor corresponding to origin PDF, for cross section reweighting
            W0 = basepdf.xfxQ(id1,x1,Q)*basepdf.xfxQ(id2,x2,Q)
            
            # Run through all PDF sets for reweighting
            for i in range(npdf):
                nrep = pdfvec[i][1]
                for j in range(nrep):

                    # Compute new PDF factor and reweight cross section
                    W = (pdfsets[i][j].xfxQ(id1,x1,Q)*pdfsets[i][j].xfxQ(id2,x2,Q))/W0

                    xsnew = xs*W
                    
                    nbin = math.floor(nbins*((np.log10(e)-np.log10(emin))/(np.log10(emax)-np.log10(emin))))
                    weights[i,stat,flavour,state,pid,nbin,j] += xsnew

    return (weights*L)/njobs

# Given the weights as a function of energy, calculates the error running over the replicas, and applying the 
# corresponding error prescription
def weights_error(weights):
    errors = np.zeros((npdf,2,3,2,4,nbins))
    for i in range(npdf):
        nrep = pdfvec[i][1]
        err_prescription = pdfvec[i][2]
        for j in range(2):
            for k in range(3):
                for l in range(2):
                    for m in range(4):
                        for n in range(nbins):
                            xs_val = weights[i][j][k][l][m][n]
                            if err_prescription == 'mc':
                                errors[i][j][k][l][m][n] += mc_error(xs_val[:nrep])
                            elif err_prescription == 'hess':
                                errors[i][j][k][l][m][n] += hess_error(xs_val[:nrep])
                            elif err_prescription == 'symhess':
                                errors[i][j][k][l][m][n] += symhess_error(xs_val[:nrep])
    return errors

###################################################################################################
###################################################################################################
                                    # PLOTTING FUNCTIONS
###################################################################################################
###################################################################################################

# Plot the total electron neutrino fluxes for all PDF sets as a function of energy
def mkplot_flux_e_total(energies,weights,errors,histname,log=False):
    pp = PdfPages(histname+'.pdf')
    ncols = 2
    nrows = 1
    fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(ncols*5,nrows*3.5))
    
    labels = [r'$FASER\nu: \nu_e + \bar{\nu}_e$', r'$SND: \nu_e + \bar{\nu}_e$', 
              r'$FASER\nu: \nu_\mu + \bar{\nu}_\mu$', r'$SND: \nu_\mu + \bar{\nu}_\mu$', 
              r'$FASER\nu: \nu_\tau + \bar{\nu}_\tau$', r'$SND: \nu_\tau + \bar{\nu}_\tau$']
    flavours = [r'$\nu_e + \bar{\nu}_e$',r'$\nu_{\mu} + \bar{\nu}_{\mu}$',r'$\nu_{\tau} + \bar{\nu}_{\tau}$']
    detectors = ['FASER'+r'$\nu$', 'SND']
    pdf = ['NNPDF3.1','CT18','MSHT','NNPDF4.0','NNPDF4.0 p.ch.']
    hadrons = [r'$\pi$', r'$K$', r'$D$', r'$\Lambda$', 'Total']
    linestyle_tuple = [('dotted', (0, (1, 1))), ('dashed', (0, (5, 5))), ('dashdotted',(0, (4, 3, 1, 3))),
                       ('densely dashdotted', (0, (3, 1, 1, 1))), ('solid', 'solid')]
    rescolors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    
    w_h = np.sum(weights[:,:,:,:,:,:,0], axis = 3)
    e_h = np.sum(errors, axis = 3)
    w = np.sum(w_h, axis = 3)
    e = np.sum(e_h, axis = 3)
    
    # Upper plot: original vs reweighted cross sections as a function of energy
    fig.suptitle('Total fluxes after reweighting to different PDF sets', fontweight='bold', fontsize=14, y=1.03) 

    p = []

    for plot, ax in enumerate(axs.flatten()):
        idet = int(plot%2)
        iflav = int(plot//2)
        for ipdf in range(npdf):
            ls_name, ls = linestyle_tuple[ipdf]
            p1 = ax.plot(np.NaN, np.NaN, ls = ls, color = rescolors[ipdf], alpha = 1)
            p2 = ax.fill(np.NaN, np.NaN, color = rescolors[ipdf], alpha=0.2)
            ax.hist(energies[:-1], energies, weights = w[ipdf][idet][iflav][:-1], histtype = 'step', color = rescolors[ipdf], ls = ls, alpha = 1)
            ax.bar(energies[:-1], height = 2*e[ipdf][idet][iflav][:-1],
                   width = energies[1:]-energies[:-1], bottom = w[ipdf][idet][iflav][:-1]-e[ipdf][idet][iflav][:-1],
                   align = 'edge', color = rescolors[ipdf], alpha = 0.2)
            p.append(p1)
            p.append(p2)

        ax.set_xscale('log')
        ax.set_xlim(10, 5000)
        if log == True: ax.set_yscale('log')
        ax.set_ylabel(r'$\phi_\nu$'+'(E) [1/bin]', fontsize=12)

        textstr = labels[plot]
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(0.05, 0.9, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)

    # if log == True:
    #     axs[0, 0].set_ylim(10**(7), 10**(11))
    #     axs[0, 1].set_ylim(10**(7), 10**(11))
    #     axs[1, 0].set_ylim(10**(7), 10**(12))
    #     axs[1, 1].set_ylim(10**(7), 10**(12))
    #     axs[2, 0].set_ylim(10**(6), 10**(9))
    #     axs[2, 1].set_ylim(10**(6), 10**(10))

    axs[0].set_xticks([10, 100, 1000, 5000])
    axs[0].set_xticklabels([r'$10^1$', r'$10^2$', r'$10^3$', r'5.0x$10^3$'])

    axs[1].set_xticks([10, 100, 1000, 5000])
    axs[1].set_xticklabels([r'$10^1$', r'$10^2$', r'$10^3$', r'5x$10^3$'])

    axs[0].set_xlabel('E (GeV)', fontsize=12)
    axs[1].set_xlabel('E (GeV)', fontsize=12)
    fig.legend([(p[0][0],p[1][0]),(p[2][0],p[3][0]),(p[4][0],p[5][0]),(p[6][0],p[7][0]),(p[8][0],p[9][0])], 
               [pdf[0],pdf[1],pdf[2],pdf[3],pdf[4]],
                frameon=True, ncol = 5, bbox_to_anchor=(0.5,1.0), loc=9, prop={'size':14})

    fig.tight_layout()
    fig.show()
    pp.savefig(fig, bbox_inches='tight')
    pp.close()

    return

# Plot the total neutrino fluxes for a given PDF set as a function of energy, with the corresponding
# contributions
def mkplot_flux_e(energies,weights,errors,pdf_final,histname,log=False):
    pp = PdfPages(histname+'.pdf')
    ncols = 2
    nrows = 3
    fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(ncols*5,nrows*3.5))
    
    labels = [r'$FASER\nu: \nu_e + \bar{\nu}_e$', r'$SND: \nu_e + \bar{\nu}_e$', 
              r'$FASER\nu: \nu_\mu + \bar{\nu}_\mu$', r'$SND: \nu_\mu + \bar{\nu}_\mu$', 
              r'$FASER\nu: \nu_\tau + \bar{\nu}_\tau$', r'$SND: \nu_\tau + \bar{\nu}_\tau$']
    flavours = [r'$\nu_e + \bar{\nu}_e$',r'$\nu_{\mu} + \bar{\nu}_{\mu}$',r'$\nu_{\tau} + \bar{\nu}_{\tau}$']
    detectors = ['FASER'+r'$\nu$', 'SND']
    hadrons = [r'$\pi$', r'$K$', r'$D$', r'$\Lambda$', 'Total']
    linestyle_tuple = [('dotted', (0, (1, 1))), ('dashed', (0, (5, 5))), ('dashdotted',(0, (4, 3, 1, 3))),
                       ('densely dashdotted', (0, (3, 1, 1, 1))), ('solid', 'solid')]
    rescolors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    
    w_h = np.sum(weights[:,:,:,:,:,0], axis = 2)
    e_h = np.sum(errors, axis = 2)
    w = np.sum(w_h, axis = 2)
    e = np.sum(e_h, axis = 2)
    
    # Upper plot: original vs reweighted cross sections as a function of energy
    fig.suptitle('Neutrino fluxes as a function of energy using '+pdf_final, fontweight='bold', fontsize=14, y=1.02) 

    p = []

    for plot, ax in enumerate(axs.flatten()):
        idet = int(plot%2)
        iflav = int(plot//2)
        for j in range(4):
            ls_name, ls = linestyle_tuple[j]
            if iflav == 0 and j == 0:
                p1 = ax.plot(np.NaN, np.NaN, ls = ls, color = rescolors[j], alpha = 1)
                p2 = ax.fill(np.NaN, np.NaN, color = rescolors[j], alpha=0.2)
                p.append(p1)
                p.append(p2)
                continue
            elif iflav == 2 and (j == 0 or j == 1 or j == 3): continue
            else:
                p1 = ax.plot(np.NaN, np.NaN, ls = ls, color = rescolors[j], alpha = 1)
                p2 = ax.fill(np.NaN, np.NaN, color = rescolors[j], alpha=0.2)
                ax.hist(energies[:-1], energies, weights = w_h[idet][iflav][j][:-1], histtype = 'step', color = rescolors[j], ls = ls, alpha = 1)
                ax.bar(energies[:-1], height = 2*e_h[idet][iflav][j][:-1],
                       width = energies[1:]-energies[:-1], bottom = w_h[idet][iflav][j][:-1]-e_h[idet][iflav][j][:-1],
                       align = 'edge', color = rescolors[j], alpha = 0.2)
                if iflav == 0:
                    p.append(p1)
                    p.append(p2)

        if iflav != 2:
            p1 = ax.plot(np.NaN, np.NaN, ls = 'solid', color = '#000000', alpha = 1)
            p2 = ax.fill(np.NaN, np.NaN, color = '#000000', alpha=0.2)
            ax.hist(energies[:-1], energies, weights = w[idet][iflav][:-1], histtype = 'step', color = '#000000', ls = 'solid', alpha = 1)
            ax.bar(energies[:-1], height = 2*e[idet][iflav][:-1],
                   width = energies[1:]-energies[:-1], bottom = w[idet][iflav][:-1]-e[idet][iflav][:-1],
                   align = 'edge', color = '#000000', alpha = 0.2)
            p.append(p1)
            p.append(p2)

        ax.set_xscale('log')
        ax.set_xlim(10, 5000)
        if log == True: ax.set_yscale('log')
        ax.set_ylabel(r'$\phi_\nu$'+'(E) [1/bin]', fontsize=12)

        textstr = labels[plot]
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(0.05, 0.9, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)

    # if log == True:
    #     axs[0, 0].set_ylim(10**(7), 10**(11))
    #     axs[0, 1].set_ylim(10**(7), 10**(11))
    #     axs[1, 0].set_ylim(10**(7), 10**(12))
    #     axs[1, 1].set_ylim(10**(7), 10**(12))
    #     axs[2, 0].set_ylim(10**(6), 10**(9))
    #     axs[2, 1].set_ylim(10**(6), 10**(10))

    axs[2, 0].set_xticks([10, 100, 1000, 5000])
    axs[2, 0].set_xticklabels([r'$10^1$', r'$10^2$', r'$10^3$', r'5.0x$10^3$'])

    axs[2, 1].set_xticks([10, 100, 1000, 5000])
    axs[2, 1].set_xticklabels([r'$10^1$', r'$10^2$', r'$10^3$', r'5x$10^3$'])

    axs[2][0].set_xlabel('E (GeV)', fontsize=12)
    axs[2][1].set_xlabel('E (GeV)', fontsize=12)
    fig.legend([(p[0][0],p[1][0]),(p[2][0],p[3][0]),(p[4][0],p[5][0]),(p[6][0],p[7][0]),(p[8][0],p[9][0])], 
               [hadrons[0],hadrons[1],hadrons[2],hadrons[3],hadrons[4]],
                frameon=True, ncol = 5, bbox_to_anchor=(0.5,1.0), loc=9, prop={'size':14})

    fig.tight_layout()
    fig.show()
    pp.savefig(fig, bbox_inches='tight')
    pp.close()

    return

# Plot the ratio of total electron neutrino fluxes wrt NNPDF3.1, for all PDF sets
def mkplot_flux_e_total_ratio(energies,weights,errors,histname,log=False):
    pp = PdfPages(histname+'.pdf')
    ncols = 2
    nrows = 1
    fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(ncols*5,nrows*3.5))
    
    labels = [r'$FASER\nu: \nu_e + \bar{\nu}_e$', r'$SND: \nu_e + \bar{\nu}_e$', 
              r'$FASER\nu: \nu_\mu + \bar{\nu}_\mu$', r'$SND: \nu_\mu + \bar{\nu}_\mu$', 
              r'$FASER\nu: \nu_\tau + \bar{\nu}_\tau$', r'$SND: \nu_\tau + \bar{\nu}_\tau$']
    flavours = [r'$\nu_e + \bar{\nu}_e$',r'$\nu_{\mu} + \bar{\nu}_{\mu}$',r'$\nu_{\tau} + \bar{\nu}_{\tau}$']
    detectors = ['FASER'+r'$\nu$', 'SND']
    pdf = ['NNPDF3.1 (ref)','CT18','MSHT','NNPDF4.0','NNPDF4.0 p.ch.']
    hadrons = [r'$\pi$', r'$K$', r'$D$', r'$\Lambda$', 'Total']
    linestyle_tuple = [('dotted', (0, (1, 1))), ('dashed', (0, (5, 5))), ('dashdotted',(0, (4, 3, 1, 3))),
                       ('densely dashdotted', (0, (3, 1, 1, 1))), ('solid', 'solid')]
    rescolors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    
    w_h = np.sum(weights[:,:,:,:,:,:,0], axis = 3)
    e_h = np.sum(errors, axis = 3)
    w = np.sum(w_h, axis = 3)
    e = np.sum(e_h, axis = 3)

    r = w[:,:,0,:]/w[0,:,0,:]
    e_r = e[:]/w[0]
    
    # Upper plot: original vs reweighted cross sections as a function of energy
    fig.suptitle('Flux ratio w.r.t. NNPDF3.1 after reweighting', fontweight='bold', fontsize=14, y=1.04) 

    p = []

    for plot, ax in enumerate(axs.flatten()):
        idet = int(plot%2)
        ls_name, ls = linestyle_tuple[0]
        if plot == 0:
            p1 = ax.plot(np.NaN, np.NaN, ls = ls, color = rescolors[0], alpha = 1)
            p2 = ax.fill(np.NaN, np.NaN, color = rescolors[0], alpha=0.2)
            p.append(p1)
            p.append(p2)
        ax.hist(energies[:-1], energies, weights = r[0][idet][:-1], histtype = 'step', color = rescolors[0], ls = ls, alpha = 1)
        ax.bar(energies[:-1], height = 2*e_r[0][idet][0][:-1],
                   width = energies[1:]-energies[:-1], bottom = r[0][idet][:-1]-e_r[0][idet][0][:-1],
                   align = 'edge', color = rescolors[0], alpha = 0.2)
        for ipdf in range(0,npdf-1):
            ls_name, ls = linestyle_tuple[ipdf+1]
            p1 = ax.plot(np.NaN, np.NaN, ls = ls, color = rescolors[ipdf+1], alpha = 1)
            p2 = ax.fill(np.NaN, np.NaN, color = rescolors[ipdf+1], alpha=0.2)
            ax.hist(energies[:-1], energies, weights = r[ipdf][idet][:-1], histtype = 'step', color = rescolors[ipdf+1], ls = ls, alpha = 1)
            ax.bar(energies[:-1], height = 2*e_r[ipdf][idet][0][:-1],
                   width = energies[1:]-energies[:-1], bottom = r[ipdf][idet][:-1]-e_r[ipdf][idet][0][:-1],
                   align = 'edge', color = rescolors[ipdf+1], alpha = 0.2)
            p.append(p1)
            p.append(p2)

        ax.set_xscale('log')
        ax.set_xlim(10, 5000)
        if log == True: ax.set_yscale('log')
        ax.set_ylabel(r'$R_{PDF/NNPDF3.1}$'+'(E)', fontsize=12)

        textstr = labels[plot]
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(0.05, 0.9, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)

    # if log == True:
    #     axs[0, 0].set_ylim(10**(7), 10**(11))
    #     axs[0, 1].set_ylim(10**(7), 10**(11))
    #     axs[1, 0].set_ylim(10**(7), 10**(12))
    #     axs[1, 1].set_ylim(10**(7), 10**(12))
    #     axs[2, 0].set_ylim(10**(6), 10**(9))
    #     axs[2, 1].set_ylim(10**(6), 10**(10))

    axs[0].set_xticks([10, 100, 1000, 5000])
    axs[0].set_xticklabels([r'$10^1$', r'$10^2$', r'$10^3$', r'5.0x$10^3$'])

    axs[1].set_xticks([10, 100, 1000, 5000])
    axs[1].set_xticklabels([r'$10^1$', r'$10^2$', r'$10^3$', r'5x$10^3$'])

    axs[0].set_ylim(0.5,1.5)
    axs[1].set_ylim(0.5,1.5)
    axs[0].set_xlabel('E (GeV)', fontsize=12)
    axs[1].set_xlabel('E (GeV)', fontsize=12)
    fig.legend([(p[0][0],p[1][0]),(p[2][0],p[3][0]),(p[4][0],p[5][0]),(p[6][0],p[7][0]),(p[8][0],p[9][0])], 
               [pdf[0],pdf[1],pdf[2],pdf[3],pdf[4]],
                frameon=True, ncol = 5, bbox_to_anchor=(0.5,1.0), loc=9, prop={'size':14})

    fig.tight_layout()
    fig.show()
    pp.savefig(fig, bbox_inches='tight')
    pp.close()

    return

# Plot the neutrino fluxes for a given PDF set as a function of pseudorapidity; discarded for the study
# def mkplot_flux_eta(weights,errors,pdf_final,histname):
#     pp = PdfPages(histname+'.pdf')
#     ncols = 2
#     nrows = 3
#     nbins = nbinsrap
#     psrap_faser = np.linspace(etamin_faser, etamax_faser, nbins)
#     psrap_snd = np.linspace(etamin_snd, etamax_snd, nbins)

#     fig, axs = plt.subplots(nrows, ncols, sharex=False, figsize=(ncols*5,nrows*3.5))
    
#     labels = [r'$FASER\nu: \nu_e + \bar{\nu}_e$', r'$SND: \nu_e + \bar{\nu}_e$', 
#               r'$FASER\nu: \nu_\mu + \bar{\nu}_\mu$', r'$SND: \nu_\mu + \bar{\nu}_\mu$', 
#               r'$FASER\nu: \nu_\tau + \bar{\nu}_\tau$', r'$SND: \nu_\tau + \bar{\nu}_\tau$']
#     flavours = [r'$\nu_e + \bar{\nu}_e$',r'$\nu_{\mu} + \bar{\nu}_{\mu}$',r'$\nu_{\tau} + \bar{\nu}_{\tau}$']
#     detectors = ['FASER'+r'$\nu$', 'SND']
#     hadrons = [r'$\pi$', r'$K$', r'$D$', r'$\Lambda$', 'Total']
#     linestyle_tuple = [
#      ('dotted', (0, (1, 1))), ('dashed', (0, (5, 5))), ('dashdotted',(0, (4, 3, 1, 3))),
#      ('densely dashdotted', (0, (3, 1, 1, 1))), ('solid', 'solid')]
#     rescolors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

#     w_h = np.sum(weights[:,:,:,:,:,0], axis = 2)
#     e_h = np.sum(errors, axis = 2)
#     w = np.sum(w_h, axis = 2)
#     e = np.sum(e_h, axis = 2)
    
#     # Upper plot: original vs reweighted cross sections as a function of energy
#     fig.suptitle('Neutrino fluxes as a function of pseudorapidity using '+pdf_final, fontweight='bold', fontsize=14, y=1.02) 

#     p = []

#     for plot, ax in enumerate(axs.flatten()):
#         idet = int(plot%2)
#         iflav = int(plot//2)
#         if idet == 0: psrap = psrap_faser
#         else: psrap = psrap_snd
#         for j in range(4):
#             ls_name, ls = linestyle_tuple[j]
#             if iflav == 0 and j == 0:
#                 p1 = ax.plot(np.NaN, np.NaN, ls = ls, color = rescolors[j], alpha = 1)
#                 p2 = ax.fill(np.NaN, np.NaN, color = rescolors[j], alpha=0.2)
#                 p.append(p1)
#                 p.append(p2)
#                 continue
#             elif iflav == 2 and (j == 0 or j == 1 or j == 3): continue
#             else:
#                 p1 = ax.plot(psrap, w_h[idet][iflav][j], ls = ls, color = rescolors[j], alpha = 1)
#                 p2 = ax.fill(np.NaN, np.NaN, color = rescolors[j], alpha=0.2)
#                 ax.fill_between(psrap, w_h[idet][iflav][j]+e_h[idet][iflav][j], w_h[idet][iflav][j]-e_h[idet][iflav][j],
#                 color = rescolors[j], alpha = 0.2)
#                 if iflav == 0:  
#                     p.append(p1)
#                     p.append(p2)

#         if iflav != 2:
#             p1 = ax.plot(psrap, w[idet][iflav], ls = 'solid', color = '#000000', alpha = 1)
#             p2 = ax.fill(np.NaN, np.NaN, color = '#000000', alpha=0.2)
#             p.append(p1)
#             p.append(p2)
#             ax.fill_between(psrap, w[idet][iflav]+e[idet][iflav], w[idet][iflav]-e[idet][iflav],
#             color = '#000000', alpha = 0.2)

#         etalims = np.array([(etamin_faser,etamax_faser),(etamin_snd,etamax_snd)])
#         ax.set_xlim(etalims[idet])
#         if iflav != 2:
#             ax.set_xticks(np.linspace(etalims[idet][0], etalims[idet][1], 5))
#             ax.set_xticklabels([' ', ' ', ' ', ' ', ' '])
#         else:
#             if idet == 0:
#                 ax.set_xticks(np.linspace(etalims[idet][0], etalims[idet][1], 5))
#                 ax.set_xticklabels(['8', '9', '10', '11', '12'])
#             else:
#                 ax.set_xticks(np.linspace(etalims[idet][0], etalims[idet][1], 5))
#                 ax.set_xticklabels(['7.0', '7.5', '8.0', '8.5', '9.0'])
        
        
#         ax.set_ylabel(r'$\phi_\nu$'+'('+r'$\eta$'+') [1/bin]', fontsize=12)

#         textstr = labels[plot]
#         props = dict(boxstyle='round', facecolor='white', alpha=1)
#         ax.text(0.05, 0.9, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)

#     axs[2][0].set_xlabel(r'$\eta$', fontsize=12)
#     axs[2][1].set_xlabel(r'$\eta$', fontsize=12)
#     fig.legend([(p[0][0],p[1][0]),(p[2][0],p[3][0]),(p[4][0],p[5][0]),(p[6][0],p[7][0]),(p[8][0],p[9][0])], 
#                [hadrons[0],hadrons[1],hadrons[2],hadrons[3],hadrons[4]],
#                 frameon=True, ncol = 5, bbox_to_anchor=(0.5,1.0), loc=9, prop={'size':14})

#     fig.tight_layout()
#     fig.show()
#     pp.savefig(fig, bbox_inches='tight')
#     pp.close()

#     return

###################################################################################################
###################################################################################################
                                    # ADDITIONAL FUNCTIONS
###################################################################################################
###################################################################################################

# Plotting of other observables, namely the flavour ratios and matter antimatter asymmetry
def ratios_flavour(weights):
    nbins = nbins1
    r = np.zeros((npdf,2,3,nbins,101))
    errors = np.zeros((npdf,2,3,nbins))

    weights_total = np.sum(np.sum(weights, axis = 3), axis = 3)

    for i in range(npdf):
        nrep = pdfvec[i][1]
        err_prescription = pdfvec[i][2]
        for j in range(2):
            for k in range(nbins):
                for l in range(nrep):
                    r0 = weights_total[i,j,0,k,l]
                    r1 = weights_total[i,j,1,k,l]
                    r2 = weights_total[i,j,2,k,l]

                    if r0 == 0:
                        if r1 == 0:
                            r[i,j,0,k,l] = 1
                        else:
                            r[i,j,0,k,l] = np.nan
                    else:
                        r[i,j,0,k,l] = r1/r0

                    if r2 == 0:
                        if r1 == 0:
                            r[i,j,1,k,l] = 1
                        else:
                            r[i,j,1,k,l] = np.NaN
                    else:
                        r[i,j,1,k,l] = r1/r2

                    if r2 == 0:
                        if r0 == 0:
                            r[i,j,2,k,l] = 1
                        else:
                            r[i,j,2,k,l] = np.nan
                    else:
                        r[i,j,2,k,l] = r0/r2
                
                for l in range(3):
                    r_val = r[i,j,l,k]
                    if err_prescription == 'mc':
                        errors[i,j,l,k] += mc_error(r_val[:nrep])
                    elif err_prescription == 'hess':
                        errors[i,j,l,k] += hess_error(r_val[:nrep])
                    elif err_prescription == 'symhess':
                        errors[i,j,l,k] += symhess_error(r_val[:nrep]) 

    return r, errors

def ratios_state(weights_):
    antimatter/matter

    pdf, detector, ratio (e, mu, tau, total), nbins, nreplica
    nbins = nbins1
    r = np.zeros((npdf,2,4,nbins,101))
    errors = np.zeros((npdf,2,4,nbins))

    weights = np.sum(weights_, axis = 4)
    weights_total = np.sum(weights, axis = 2)

    for i in range(npdf):
        nrep = pdfvec[i][1]
        err_prescription = pdfvec[i][2]
        for j in range(2): #detector
            for k in range(nbins):
                for l in range(nrep):
                    for m in range(3): #flavour
                        r0 = weights[i,j,m,0,k,l]
                        r1 = weights[i,j,m,1,k,l]
                        if r0 == 0:
                            if r1 == 0:
                                r[i,j,m,k,l] = 1
                            else:
                                r[i,j,m,k,l] = np.nan
                        else:
                            r[i,j,m,k,l] = r1/r0
                    
                    r[i,j,3,k,l] = weights_total[i,j,1,k,l]/weights_total[i,j,0,k,l]
                    
                for l in range(4):
                    r_val = r[i,j,l,k]
                    if err_prescription == 'mc':
                        errors[i,j,l,k] += mc_error(r_val[:nrep])
                    elif err_prescription == 'hess':
                        errors[i,j,l,k] += hess_error(r_val[:nrep])
                    elif err_prescription == 'symhess':
                        errors[i,j,l,k] += symhess_error(r_val[:nrep]) 

    return r, errors

def mkplot_ratio_flavour(energies,ratios,errors,histname,eset='mc',log=False):
    pp = PdfPages(histname+'_'+eset+'.pdf')
    ncols = 2
    nrows = 3
    eps = 0.2
    fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(ncols*5,nrows*3.5))
    ref = ratios[0, :, :, :, 0]
    if eset == 'mc':
        r = ratios[3:, :, :, :, 0]
        e = errors[3:, :, :, :]
        setname = 'Monte Carlo'
        pdfset = ['NNPDF4.0','NNPDF4.0 P.Ch.']
    elif eset == 'hess':
        r = ratios[1:3, :, :, :, 0]
        e = errors[1:3, :, :, :]
        setname = 'Hessian'
        pdfset = ['CT18','MSHT']

    tags = [r'$\nu_{\mu}/\nu_e$',r'$\nu_{\mu}/\nu_{\tau}$',r'$\nu_e/\nu_{\tau}$']
    detectors = ['FASER'+r'$\nu$', 'SND']
    linestyle_tuple = [
     ('dotted', (0, (1, 1))), ('dashed', (0, (5, 5))), ('dashdotted',(0, (4, 3, 1, 3))),
     ('dashdotdotted', (0, (3, 3, 1, 3, 1, 3))), ('solid', 'solid')]
    rescolors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    
   Upper plot: original vs reweighted cross sections as a function of energy
    fig.suptitle('Flavour asymmetry for '+setname+' sets\n as a function of energy (normalized to NNPDF3.1)', fontweight='bold', fontsize=14, y=1.06)

    p = []

    for plot, ax in enumerate(axs.flatten()):
        idet = int(plot%2)
        iflav = int(plot//2)
        refval = ref[idet][iflav]
        for ipdf in range(2):
            ls_name, ls = linestyle_tuple[ipdf]
            p1 = ax.plot(energies, np.divide(r[ipdf][idet][iflav], refval), ls = ls, color = rescolors[ipdf], alpha = 1)
            p2 = ax.fill(np.NaN, np.NaN, color = rescolors[ipdf], alpha=0.2)
            ax.fill_between(energies, np.divide(r[ipdf][idet][iflav]+e[ipdf][idet][iflav], refval), np.divide(r[ipdf][idet][iflav]-e[ipdf][idet][iflav], refval),
            color = rescolors[ipdf], alpha = 0.2)
            p.append(p1)
            p.append(p2)

        if log == True:
            ax.set_yscale('log')

        ax.set_xscale('log')
        ax.set_xlim(emin, emax)
        ax.set_ylim(1-eps,1+eps)
        ax.set_ylabel(tags[iflav], fontsize=12)

        textstr = tags[iflav] + ' at ' + detectors[idet]
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(0.05, 0.9, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)

    fig.legend([(p[0][0],p[1][0]),(p[2][0],p[3][0])],[pdfset[0],pdfset[1]],
                frameon=True, ncol=2, bbox_to_anchor=(0.5,1.0), loc=9, prop={'size':14})

    axs[2][0].set_xlabel('E (GeV)', fontsize=12)
    axs[2][1].set_xlabel('E (GeV)', fontsize=12)

    fig.tight_layout()
    fig.show()
    pp.savefig(fig, bbox_inches='tight')
    pp.close()

    return

def mkplot_ratio_state(energies,ratios,errors,histname,eset='mc',log=False): 
    pp = PdfPages(histname+'_'+eset+'.pdf')
    ncols = 2
    nrows = 4
    eps = 0.2
    fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(ncols*5,nrows*3.5))
    ref = ratios[0, :, :, :, 0]
    if eset == 'mc':
        r = ratios[3:, :, :, :, 0]
        e = errors[3:, :, :, :]
        setname = 'Monte Carlo'
        pdfset = ['NNPDF4.0','NNPDF4.0 P.Ch.']
    elif eset == 'hess':
        r = ratios[1:3, :, :, :, 0]
        e = errors[1:3, :, :, :]
        setname = 'Hessian'
        pdfset = ['CT18','MSHT']

    tags = [r'$\bar{\nu}_e/\nu_e$',r'$\bar{\nu}_{\mu}/\nu_{\mu}$',r'$\bar{\nu}_{\tau}/\nu_{\tau}$',r'$\bar{\nu}/\nu$']
    detectors = ['FASER'+r'$\nu$', 'SND']
    linestyle_tuple = [('dotted', (0, (1, 1))), ('dashed', (0, (5, 5))), ('dashdotted',(0, (4, 3, 1, 3))),
                       ('densely dashdotted', (0, (3, 1, 1, 1))), ('solid', 'solid')]
    rescolors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    
   Upper plot: original vs reweighted cross sections as a function of energy
    fig.suptitle('Matter-antimatter for '+setname+' sets\n as a function of energy (normalized to NNPDF3.1)', fontweight='bold', fontsize=14, y=1.06)

    p = []

    for plot, ax in enumerate(axs.flatten()):
        idet = int(plot%2)
        iflav = int(plot//2)
        refval = ref[idet][iflav]
        for ipdf in range(2):
            ls_name, ls = linestyle_tuple[ipdf]
            p1 = ax.plot(energies, np.divide(r[ipdf][idet][iflav], refval), ls = ls, color = rescolors[ipdf], alpha = 1)
            p2 = ax.fill(np.NaN, np.NaN, color = rescolors[ipdf], alpha=0.2)
            ax.fill_between(energies, np.divide(r[ipdf][idet][iflav]+e[ipdf][idet][iflav], refval), np.divide(r[ipdf][idet][iflav]-e[ipdf][idet][iflav], refval),
            color = rescolors[ipdf], alpha = 0.2)
            p.append(p1)
            p.append(p2)

        if log == True:
            ax.set_yscale('log')

        ax.set_xscale('log')
        ax.set_xlim(emin, emax)
        ax.set_ylim(1-eps,1+eps)
        ax.set_ylabel(tags[iflav], fontsize=12)

        textstr = tags[iflav] + ' at ' + detectors[idet]
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(0.05, 0.9, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)

    fig.legend([(p[0][0],p[1][0]),(p[2][0],p[3][0])],[pdfset[0],pdfset[1]],
                frameon=True, ncol=2, bbox_to_anchor=(0.5,1.0), loc=9, prop={'size':14})

    axs[3][0].set_xlabel('E (GeV)', fontsize=12)
    axs[3][1].set_xlabel('E (GeV)', fontsize=12)

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

weights = reweight_flux()
errors = weights_error(weights)
energies = np.logspace(np.log10(emin),np.log10(emax),nbins)
# for j in range(npdf):
#     pdf = pdfvec[j][0]
#     mkplot_flux_e(energies,weights[j],errors[j],pdf,'reweighting_'+pdf+'_e_final')
    #mkplot_flux_eta(weights_rap[j],errors_rap[j],pdf,'reweighting_'+pdf+'_ps_test')
mkplot_flux_e_total(energies,weights,errors,'reweighting_comparison')
mkplot_flux_e_total_ratio(energies,weights,errors,'flux_ratios')