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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.integrate as integrate

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

# Function that reads the replica weights given a PDF set and a replica number
def weight_read(ipdf, iflav, error):
    weights_out = np.zeros(pdfvec[ipdf][1]-1)
    with open('/data/theorie/josepsolac/pythia8307/examples/weights_def/weights_'+pdfvec[ipdf][0]+'_'+error+'.dat') as f:
        for k,line in enumerate(f):
            weights = line.split(' ')
            if weights[0] == 'R_'+str(iflav)+'_0': 
                for irep in range(1,pdfvec[ipdf][1]):
                    weights_out[irep-1] += float(weights[irep])
                break
    return weights_out

###################################################################################################
###################################################################################################
                                # PARAMETERS AND CONSTRAINTS
###################################################################################################
###################################################################################################

# PDF sets of the study
pdfvec = [['NNPDF31_nnlo_as_0118', 101, 'mc'],['NNPDF40_nnlo_as_01180', 101, 'mc'],['NNPDF40_nnlo_pch_as_01180', 101, 'mc'],['NNPDF31_nnlo_as_0118_1000', 1001, 'mc']]


# Establish parameters of the data analysis
npdf = len(pdfvec)
nratio = 2
xbins = 1000

sqrts = 13000
s = sqrts**2.
# x bins depending on the desired scale of the resulting plots
x_val = np.linspace(0.00001,1,xbins)
#x_val = np.logspace(-8,0,xbins)
errorvec = ['1','2','5','10']

qmin = 2
qmax = 12
qbins = 50
q_val = np.linspace(qmin,qmax,qbins)
epsrel=1e-3

# Call all PDF sets and replicas
pdfsets = []
for jpdf in range(npdf):
    pdfsets.append(callpdf(pdfvec[jpdf][0], pdfvec[jpdf][1]))

###################################################################################################
###################################################################################################
                            # COMPUTATION FUNCTIONS (REWEIGHT AND ERROR)
###################################################################################################
###################################################################################################

# Compute the unweighted and reweighted PDFs (charm by default) at a certain Q scale (1.65 GeV by default)
def charm_pdf(iflav, error, flavour = 4, Q = 1.65):
    weights = np.zeros((npdf,1000))
    for jpdf in range(npdf):
        weights[jpdf][:pdfvec[jpdf][1]-1] += weight_read(jpdf, iflav, error)
    xc_uw = np.zeros((2,npdf,xbins))
    xc_rw = np.zeros((2,npdf,xbins))
    for ipdf in range(npdf):
        nreps = pdfvec[ipdf][1]
        for ix,x in enumerate(x_val):
            xc_i = np.zeros(nreps-1)

            for irep in range(1,nreps):
                xc_i[irep-1] = pdfsets[ipdf][irep].xfxQ(flavour,x,Q) + pdfsets[ipdf][irep].xfxQ(-flavour,x,Q)
            xc_uw[0,ipdf,ix] += np.mean(xc_i)
            xc_uw[1,ipdf,ix] += 1.642*np.std(xc_i)
            xc_mean = 0.

            for irep in range(nreps-1):
                xc_mean += weights[ipdf][irep]*xc_i[irep]/(nreps-1)
            xc_rw[0,ipdf,ix] += xc_mean
            xc_std = 0.

            for irep in range(nreps-1):
                xc_std += ((xc_i[irep]-xc_mean)**2.)*weights[ipdf][irep]
            xc_rw[1][ipdf][ix] += 1.642*math.sqrt(xc_std/(nreps-1))

    return xc_uw, xc_rw

# Compute the unweighted and reweighted gg luminosities at a certain sqrt(s) scale (13 TeV by default)
def charm_gglumi(iflav, error):
    weights = np.zeros((npdf,1000))
    for jpdf in range(npdf):
        weights[jpdf][:pdfvec[jpdf][1]-1] += weight_read(jpdf, iflav, error)
    lumi_uw = np.zeros((2,npdf,qbins))
    lumi_rw = np.zeros((2,npdf,qbins))
    for ipdf in range(npdf):
        nreps = pdfvec[ipdf][1]
        for iq,q in enumerate(q_val):
            print(q)
            tau = q**2./s
            lumi_uw_i = np.zeros(nreps-1)
            lumi_rw_i = np.zeros(nreps-1)

            if(tau < 0 or tau > 1):
                    print("invalid value of tau = ",tau)
                    exit()

            for irep in range(1,nreps):
                w = weights[ipdf][irep-1]
                I = integrate.quad(lambda y: (pdfsets[ipdf][irep].xfxQ(0,y,q)*pdfsets[ipdf][irep].xfxQ(0,tau/y,q))/(y*tau), tau, 1, epsrel=epsrel)[0]
                lumi_uw_i[irep-1] += I
                lumi_rw_i[irep-1] += I*w

            lumi_uw[0][ipdf][iq] += np.mean(lumi_uw_i)
            lumi_uw[1][ipdf][iq] += 1.642*np.std(lumi_uw_i)

            lumi_rw[0][ipdf][iq] += np.mean(lumi_rw_i)
            lumi_std = 0
            for irep in range(nreps-1):
                lumi_std += ((lumi_uw_i[irep]-lumi_rw[0][ipdf][iq])**2.)*weights[ipdf][irep]
            lumi_rw[1][ipdf][iq] += 1.642*math.sqrt(lumi_std/(nreps-1))
        
    return lumi_uw/s, lumi_rw/s

# Compute the unweighted and reweighted gc luminosities at a certain sqrt(s) scale (13 TeV by default)
def charm_gclumi(iflav, error):
    weights = np.zeros((npdf,1000))
    for jpdf in range(npdf):
        weights[jpdf][:pdfvec[jpdf][1]-1] += weight_read(jpdf, iflav, error)
    lumi_uw = np.zeros((2,npdf,qbins))
    lumi_rw = np.zeros((2,npdf,qbins))
    for ipdf in range(npdf):
        nreps = pdfvec[ipdf][1]
        for iq,q in enumerate(q_val):
            print(q)
            tau = q**2./s
            lumi_uw_i = np.zeros(nreps-1)
            lumi_rw_i = np.zeros(nreps-1)
            if(tau < 0 or tau > 1):
                    print("invalid value of tau = ",tau)
                    exit()
            for irep in range(1,nreps):
                w = weights[ipdf][irep-1]
                I = integrate.quad(lambda y: ((pdfsets[ipdf][irep].xfxQ(4,y,q)+pdfsets[ipdf][irep].xfxQ(-4,y,q))*
                                               pdfsets[ipdf][irep].xfxQ(0,tau/y,q))/(y*tau), tau, 1, epsrel=epsrel)[0]
                lumi_uw_i[irep-1] += I
                lumi_rw_i[irep-1] += I*w
                # lumi_uw_i[irep-1] += integrate.quad(integrand_gglumi(y, q, ipdf, irep, 1), tau, 1, epsrel=epsrel)[0]
                # lumi_rw_i[irep-1] += integrate.quad(integrand_gglumi(y, q, ipdf, irep, weights[ipdf][irep-1]), tau, 1, epsrel=epsrel)[0]

            lumi_uw[0][ipdf][iq] += np.mean(lumi_uw_i)
            lumi_uw[1][ipdf][iq] += 1.642*np.std(lumi_uw_i)

            lumi_rw[0][ipdf][iq] += np.mean(lumi_rw_i)
            lumi_std = 0
            for irep in range(nreps-1):
                lumi_std += ((lumi_uw_i[irep]-lumi_rw[0][ipdf][iq])**2.)*weights[ipdf][irep]
            lumi_rw[1][ipdf][iq] += 1.642*math.sqrt(lumi_std/(nreps-1))
        
    return lumi_uw/s, lumi_rw/s

# Compute the unweighted and reweighted gq luminosities at a certain sqrt(s) scale (13 TeV by default)
def charm_gqlumi(iflav, error):
    weights = np.zeros((npdf,1000))
    for jpdf in range(npdf):
        weights[jpdf][:pdfvec[jpdf][1]-1] += weight_read(jpdf, iflav, error)
    lumi_uw = np.zeros((2,npdf,qbins))
    lumi_rw = np.zeros((2,npdf,qbins))
    for ipdf in range(npdf):
        nreps = pdfvec[ipdf][1]
        for iq,q in enumerate(q_val):
            print(q)
            tau = q**2./s
            lumi_uw_i = np.zeros(nreps-1)
            lumi_rw_i = np.zeros(nreps-1)
            if(tau < 0 or tau > 1):
                    print("invalid value of tau = ",tau)
                    exit()
            for irep in range(1,nreps):
                w = weights[ipdf][irep-1]
                I = integrate.quad(lambda y: ((pdfsets[ipdf][irep].xfxQ(1,y,q)+pdfsets[ipdf][irep].xfxQ(-1,y,q)+
                                               pdfsets[ipdf][irep].xfxQ(2,y,q)+pdfsets[ipdf][irep].xfxQ(-2,y,q)+
                                               pdfsets[ipdf][irep].xfxQ(3,y,q)+pdfsets[ipdf][irep].xfxQ(-3,y,q)+
                                               pdfsets[ipdf][irep].xfxQ(4,y,q)+pdfsets[ipdf][irep].xfxQ(-4,y,q))*
                                               pdfsets[ipdf][irep].xfxQ(0,tau/y,q))/(y*tau), tau, 1, epsrel=epsrel)[0]
                lumi_uw_i[irep-1] += I
                lumi_rw_i[irep-1] += I*w
                # lumi_uw_i[irep-1] += integrate.quad(integrand_gglumi(y, q, ipdf, irep, 1), tau, 1, epsrel=epsrel)[0]
                # lumi_rw_i[irep-1] += integrate.quad(integrand_gglumi(y, q, ipdf, irep, weights[ipdf][irep-1]), tau, 1, epsrel=epsrel)[0]

            lumi_uw[0][ipdf][iq] += np.mean(lumi_uw_i)
            lumi_uw[1][ipdf][iq] += 1.642*np.std(lumi_uw_i)

            lumi_rw[0][ipdf][iq] += np.mean(lumi_rw_i)
            lumi_std = 0
            for irep in range(nreps-1):
                lumi_std += ((lumi_uw_i[irep]-lumi_rw[0][ipdf][iq])**2.)*weights[ipdf][irep]
            lumi_rw[1][ipdf][iq] += 1.642*math.sqrt(lumi_std/(nreps-1))
        
    return lumi_uw/s, lumi_rw/s

# Provide percentage of constraint above x = 0.5, being defined as reweighted area of error divided by
# unweighted area of error
def rel_constraint_pdf(err_old, err_new):
    dx = 1/xbins
    R = np.zeros(npdf)
    for ipdf in range(npdf):
        I_old = 0.
        I_new = 0.
        for ix,x in enumerate(x_val[int(xbins/2):]):
            ix += int(xbins/2)
            I_old += 2*err_old[ipdf][ix]*dx
            I_new += 2*err_new[ipdf][ix]*dx
        R[ipdf] += I_new/I_old
    return R

# Provide percentage of constraint from Q = 2 GeV to Q = 12 GeV, being defined as reweighted area of error 
# divided by unweighted area of error
def rel_constraint_lumi(err_old, err_new):
    dq = 1/qbins
    R = np.zeros(npdf)
    for ipdf in range(npdf):
        I_new = 0.
        I_old = 0.
        for i,iq in enumerate(q_val):
            I_old += 2*err_old[ipdf][i]*dq
            I_new += 2*err_new[ipdf][i]*dq
        R[ipdf] += I_new/I_old
    return R

###################################################################################################
###################################################################################################
                                    # PLOTTING FUNCTIONS
###################################################################################################
###################################################################################################

# Plots unweighted vs reweighted PDFs and corresponding errors
def mkplot_charm_pdf(iflav,ratios_uw,ratios_rw,error,R,log = False,Q=1.65):
    if log == True:
        fileend = '_log.pdf'
    else:
        fileend = '.pdf'
    
    pp = PdfPages('/data/theorie/josepsolac/pythia8307/examples/PDF plots default/PDF/charmPDF_'+error+'_'+str(iflav+1)+fileend)
    ncols = 1
    nrows = npdf-1
    fig, axs = plt.subplots(nrows, ncols, sharex=False, figsize=(ncols*5,nrows*3.5))
    tag = r'$\nu_{\tau}/\nu_e$'
    pdfset = ['NNPDF3.1','NNPDF4.0','NNPDF4.0 p.ch.','NNPDF3.1 1000r']#
    legend = ['PDF','PDF + FPF']
    detectors = ['FASER'+r'$\nu$', 'SND']
    linestyle_tuple = [('dotted', (0, (1, 1))), ('dashed', (0, (5, 5))), ('dashdotted',(0, (4, 3, 1, 3))),
                       ('dashdotdotted', (0, (3, 3, 1, 3, 1, 3))), ('solid', 'solid')]
    rescolors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    
   # Upper plot: original vs reweighted cross sections as a function of energy
    fig.suptitle('Gluon PDF constraint @ Q = '+str(Q)+' GeV and '+r'$\sigma_{exp}$ = ' + error + '%', fontweight='bold', fontsize=14, y=1.03)
    
    r = np.array([ratios_uw[0][1:],ratios_rw[0][1:]])
    e = np.array([ratios_uw[1][1:],ratios_rw[1][1:]])

    p = []

    for plot, ax in enumerate(axs.flatten()):
        ax2 = inset_axes(axs[plot], width = '50%', height = '25%', loc = 'lower left', bbox_to_anchor=(0.4,0.07,1.1,1.1), bbox_transform=axs[plot].transAxes)
        ax2.set_title('Error band width', fontsize = 6, pad = 2)
        for ir in range(2):
            ls_name, ls = linestyle_tuple[ir]
            p1 = ax.plot(np.NaN, np.NaN, ls = ls, color = rescolors[ir], alpha = 1)
            p2 = ax.fill(np.NaN, np.NaN, color = rescolors[ir], alpha=0.2)
            ax.plot(x_val, r[ir][plot], ls = ls, color = rescolors[ir], alpha = 1)
            ax.fill_between(x_val, r[ir][plot]+e[ir][plot], r[ir][plot]-e[ir][plot], color = rescolors[ir], alpha=0.2)
            p.append(p1)
            p.append(p2)
            ax2.plot(x_val, 2*e[ir][plot], ls = ls, color = rescolors[ir], alpha = 1)
        if log == True:
            ax.set_xscale('log')
            ax2.set_xscale('log')
            ax.set_xlim(10**(-8),1)
            ax2.set_xlim(10**(-8),1)
        else: 
            ax.set_xlim(0,1)
            ax2.set_xlim(0,1)

        #ax2.set_xticks(fontsize = 6)
        #ax2.set_yticks(fontsize = 6)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel(r'$xc^{+}(x,Q)$', fontsize=12)
        ax2.set_xlabel('x', fontsize=6, labelpad = -2)
        ax2.set_ylabel('90% C.L.', fontsize=6)
        ax2.tick_params(axis='both', which='both', labelsize=4)
        if plot == 0:
            ax2.set_ylim(0, 0.03)
            ax2.set_yticks([0,0.015,0.03])
            ax2.set_yticklabels(['0.000', '0.015', '0.030'])
        elif plot == 1:
            ax2.set_ylim(0, 0.00002)
            if log == False:
                ax.set_xlim(0.5, 1)
                ax2.set_xlim(0.5, 1)
            ax2.set_yticks([0,0.00001,0.00002])
            ax2.set_yticklabels(['0', r'$10^{-5}$', r'$2x10^{-5}$'])
        elif plot == 2:
            ax2.set_ylim(0, 0.04)
            ax2.set_yticks([0,0.02,0.04])
            ax2.set_yticklabels(['0.00', '0.02', '0.04'])

        textstr = pdfset[plot+1]
        
        props1 = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(0.05, 0.90, textstr, transform=ax.transAxes, fontsize=10, va='bottom', bbox=props1)

        props2 = dict(boxstyle='round', facecolor='white', alpha=1)
        if round(R[plot+1],2) >= 1:
            ax.text(0.95, 0.90, 'No constraint', transform=ax.transAxes, fontsize=10, va='bottom', ha='right', bbox=props2)
        else:
            ax.text(0.95, 0.90, str(int(100*round(R[plot+1],2)))+'%'+' constraint', transform=ax.transAxes, fontsize=10, va='bottom', ha='right', bbox=props2)

    fig.legend([(p[0][0],p[1][0]),(p[2][0],p[3][0])],[legend[0],legend[1]],
                frameon=True, ncol=2, bbox_to_anchor=(0.5,1.01), loc=9, prop={'size':14})

    props3 = dict(boxstyle='round', facecolor='white', alpha=1)
    axs[0].text(0.995, 1.025, r'$Q = 1.65$'+' GeV', transform=axs[0].transAxes, fontsize=8, va='bottom', ha='right', bbox=props3)

    props4 = dict(boxstyle='round', facecolor='white', alpha=1)
    axs[0].text(0.005, 1.025, r'$\sigma_{exp}$ = ' + error + '%', transform=axs[0].transAxes, fontsize=8, va='bottom', ha='left', bbox=props4)
    
    axs[0].set_title(tag + ' at ' + detectors[iflav])
    if log==True:
        axs[0].set_ylim(-5,25)
        axs[1].set_ylim(-5,10)
        axs[2].set_ylim(-5,15)
    else:
        axs[0].set_ylim(-0.05,0.05)
        axs[1].set_ylim(-0.0002,0.0002)
        axs[2].set_ylim(-0.05,0.05)
        axs[1].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
    fig.tight_layout()
    fig.show()
    pp.savefig(fig, bbox_inches='tight')
    pp.close()

    return

# Plots unweighted vs reweighted gg luminosity and corresponding errors
def mkplot_charm_gglumi(iflav,lumi_uw,lumi_rw,error,R,log = False):
    fileend = '.pdf'

    pp = PdfPages('/data/theorie/josepsolac/pythia8307/examples/PDF plots default/gg lumi/charm_gglumi_'+error+'_'+str(iflav+1)+fileend)
    ncols = 1
    nrows = npdf-1
    fig, axs = plt.subplots(nrows, ncols, sharex=False, figsize=(ncols*5,nrows*3.5))
    tag = r'$\nu_{\tau}/\nu_e$'
    pdfset = ['NNPDF3.1','NNPDF4.0','NNPDF4.0 p.ch.','NNPDF3.1 1000r']
    legend = ['PDF','PDF + FPF']
    detectors = ['FASER'+r'$\nu$', 'SND']
    linestyle_tuple = [('dotted', (0, (1, 1))), ('dashed', (0, (5, 5))), ('dashdotted',(0, (4, 3, 1, 3))),
                       ('dashdotdotted', (0, (3, 3, 1, 3, 1, 3))), ('solid', 'solid')]
    rescolors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    
   # Upper plot: original vs reweighted cross sections as a function of energy
    fig.suptitle(r'$\mathcal{L}_{gg}$'' constraint for '+r'$\sigma_{exp}$ = ' + error + '%' + ' at ' + r'$\sqrt{s} = 13$'+' TeV\n Normalized to NNPDF3.1', fontweight='bold', fontsize=14, y=1.03)
    
    r = np.array([lumi_uw[0][1:]/lumi_uw[0][1:],lumi_rw[0][1:]/lumi_uw[0][1:]])
    e = np.array([lumi_uw[1][1:]/lumi_uw[0][1:],lumi_rw[1][1:]/lumi_uw[0][1:]])

    p = []

    for plot, ax in enumerate(axs.flatten()):
        ax2 = inset_axes(axs[plot], width = '50%', height = '25%', loc = 'lower left', bbox_to_anchor=(0.4,0.07,1.1,1.1), bbox_transform=axs[plot].transAxes)
        ax2.set_title('Error band width', fontsize = 6, pad = 2)
        for ir in range(2):
            ls_name, ls = linestyle_tuple[ir]
            p1 = ax.plot(np.NaN, np.NaN, ls = ls, color = rescolors[ir], alpha = 1)
            p2 = ax.fill(np.NaN, np.NaN, color = rescolors[ir], alpha=0.2)
            ax.plot(q_val, r[ir][plot], ls = ls, color = rescolors[ir], alpha = 1)
            ax.fill_between(q_val, r[ir][plot]+e[ir][plot], r[ir][plot]-e[ir][plot], color = rescolors[ir], alpha=0.2)
            p.append(p1)
            p.append(p2)
            ax2.plot(q_val, 2*e[ir][plot], ls = ls, color = rescolors[ir], alpha = 1)
        
        ax.set_xlim(2,12)
        ax2.set_xlim(2,12)
        ax.set_ylim(0.5,1.5)
        ax.set_xlabel('Q (GeV)', fontsize=12)
        ax.set_ylabel(r'$\mathcal{L}_{gg}/\mathcal{L}_{gg}(ref)$', fontsize=12)
        ax2.set_xlabel('Q (GeV)', fontsize=6, labelpad = -2)
        ax2.set_ylabel('90% C.L.', fontsize=6)
        ax2.tick_params(axis='both', which='both', labelsize=4)

        textstr = pdfset[plot+1]
        
        props1 = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(0.05, 0.90, textstr, transform=ax.transAxes, fontsize=10, va='bottom', bbox=props1)

        props2 = dict(boxstyle='round', facecolor='white', alpha=1)
        if round(R[plot+1],2) >= 1:
            ax.text(0.95, 0.90, 'No constraint', transform=ax.transAxes, fontsize=10, va='bottom', ha='right', bbox=props2)
        else:
            ax.text(0.95, 0.90, str(int(100*round(R[plot+1],2)))+'%'+' constraint', transform=ax.transAxes, fontsize=10, va='bottom', ha='right', bbox=props2)

    fig.legend([(p[0][0],p[1][0]),(p[2][0],p[3][0])],[legend[0],legend[1]],
                frameon=True, ncol=2, bbox_to_anchor=(0.5,0.985), loc=9, prop={'size':14})
    axs[0].set_title(tag + ' at ' + detectors[iflav])

    props3 = dict(boxstyle='round', facecolor='white', alpha=1)
    axs[0].text(0.995, 1.025, r'$\sqrt{s} = 13$'+' TeV', transform=axs[0].transAxes, fontsize=8, va='bottom', ha='right', bbox=props3)

    props4 = dict(boxstyle='round', facecolor='white', alpha=1)
    axs[0].text(0.005, 1.025, r'$\sigma_{exp}$ = ' + error + '%', transform=axs[0].transAxes, fontsize=8, va='bottom', ha='left', bbox=props4)

    fig.tight_layout()
    fig.show()
    pp.savefig(fig, bbox_inches='tight')
    pp.close()

    return

# Plots unweighted vs reweighted gq luminosity and corresponding errors
def mkplot_charm_gqlumi(iflav,lumi_uw,lumi_rw,error,R):
    fileend = '.pdf'

    pp = PdfPages('/data/theorie/josepsolac/pythia8307/examples/PDF plots default/gq lumi/charm_gqlumi_'+error+'_'+str(iflav+1)+fileend)
    ncols = 1
    nrows = npdf-1
    fig, axs = plt.subplots(nrows, ncols, sharex=False, figsize=(ncols*5,nrows*3.5))
    tag = r'$\nu_{\tau}/\nu_e$'
    pdfset = ['NNPDF3.1','NNPDF4.0','NNPDF4.0 p.ch.','NNPDF3.1 1000r']
    legend = ['PDF','PDF + FPF']
    detectors = ['FASER'+r'$\nu$', 'SND']
    linestyle_tuple = [('dotted', (0, (1, 1))), ('dashed', (0, (5, 5))), ('dashdotted',(0, (4, 3, 1, 3))),
                       ('dashdotdotted', (0, (3, 3, 1, 3, 1, 3))), ('solid', 'solid')]
    rescolors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    
   # Upper plot: original vs reweighted cross sections as a function of energy
    fig.suptitle(r'$\mathcal{L}_{gq}$'' constraint for '+r'$\sigma_{exp}$ = ' + error + '%' + ' at ' + r'$\sqrt{s} = 13$'+' TeV\n Normalized to NNPDF3.1', fontweight='bold', fontsize=14, y=1.03)
    
    r = np.array([lumi_uw[0][1:]/lumi_uw[0][1:],lumi_rw[0][1:]/lumi_uw[0][1:]])
    e = np.array([lumi_uw[1][1:]/lumi_uw[0][1:],lumi_rw[1][1:]/lumi_uw[0][1:]])

    p = []

    for plot, ax in enumerate(axs.flatten()):
        ax2 = inset_axes(axs[plot], width = '50%', height = '25%', loc = 'lower left', bbox_to_anchor=(0.4,0.07,1.1,1.1), bbox_transform=axs[plot].transAxes)
        ax2.set_title('Error band width', fontsize = 6, pad = 2)
        for ir in range(2):
            ls_name, ls = linestyle_tuple[ir]
            p1 = ax.plot(np.NaN, np.NaN, ls = ls, color = rescolors[ir], alpha = 1)
            p2 = ax.fill(np.NaN, np.NaN, color = rescolors[ir], alpha=0.2)
            ax.plot(q_val, r[ir][plot], ls = ls, color = rescolors[ir], alpha = 1)
            ax.fill_between(q_val, r[ir][plot]+e[ir][plot], r[ir][plot]-e[ir][plot], color = rescolors[ir], alpha=0.2)
            p.append(p1)
            p.append(p2)
            ax2.plot(q_val, 2*e[ir][plot], ls = ls, color = rescolors[ir], alpha = 1)
        ax.set_xlim(2,12)
        ax2.set_xlim(2,12)
        ax.set_ylim(0.5,1.5)
        ax.set_xlabel('Q (GeV)', fontsize=12)
        ax.set_ylabel(r'$\mathcal{L}_{gq}/\mathcal{L}_{gq}(ref)$', fontsize=12)
        ax2.set_xlabel('Q (GeV)', fontsize=6, labelpad = -2)
        ax2.set_ylabel('90% C.L.', fontsize=6)
        ax2.tick_params(axis='both', which='both', labelsize=4)

        textstr = pdfset[plot+1]
        
        props1 = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(0.05, 0.90, textstr, transform=ax.transAxes, fontsize=10, va='bottom', bbox=props1)

        props2 = dict(boxstyle='round', facecolor='white', alpha=1)
        if round(R[plot+1],2) >= 1:
            ax.text(0.95, 0.90, 'No constraint', transform=ax.transAxes, fontsize=10, va='bottom', ha='right', bbox=props2)
        else:
            ax.text(0.95, 0.90, str(int(100*round(R[plot+1],2)))+'%'+' constraint', transform=ax.transAxes, fontsize=10, va='bottom', ha='right', bbox=props2)

    fig.legend([(p[0][0],p[1][0]),(p[2][0],p[3][0])],[legend[0],legend[1]],
                frameon=True, ncol=2, bbox_to_anchor=(0.5,0.985), loc=9, prop={'size':14})

    props3 = dict(boxstyle='round', facecolor='white', alpha=1)
    axs[0].text(0.995, 1.025, r'$\sqrt{s} = 13$'+' TeV', transform=axs[0].transAxes, fontsize=8, va='bottom', ha='right', bbox=props3)

    props4 = dict(boxstyle='round', facecolor='white', alpha=1)
    axs[0].text(0.005, 1.025, r'$\sigma_{exp}$ = ' + error + '%', transform=axs[0].transAxes, fontsize=8, va='bottom', ha='left', bbox=props4)

    axs[0].set_title(tag + ' at ' + detectors[iflav])
    
    fig.tight_layout()
    fig.show()
    pp.savefig(fig, bbox_inches='tight')
    pp.close()

    return

# Plots unweighted vs reweighted gc luminosity and corresponding errors
def mkplot_charm_gclumi(iflav,lumi_uw,lumi_rw,error,R):
    fileend = '.pdf'

    pp = PdfPages('/data/theorie/josepsolac/pythia8307/examples/PDF plots default/gc lumi/charm_gclumi_'+error+'_'+str(iflav+1)+fileend)
    ncols = 1
    nrows = npdf-1
    fig, axs = plt.subplots(nrows, ncols, sharex=False, figsize=(ncols*5,nrows*3.5))
    tag = r'$\nu_{\tau}/\nu_e$'
    pdfset = ['NNPDF3.1','NNPDF4.0','NNPDF4.0 p.ch.','NNPDF3.1 1000r']
    legend = ['PDF','PDF + FPF']
    detectors = ['FASER'+r'$\nu$', 'SND']
    linestyle_tuple = [('dotted', (0, (1, 1))), ('dashed', (0, (5, 5))), ('dashdotted',(0, (4, 3, 1, 3))),
                       ('dashdotdotted', (0, (3, 3, 1, 3, 1, 3))), ('solid', 'solid')]
    rescolors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    
   # Upper plot: original vs reweighted cross sections as a function of energy
    fig.suptitle(r'$\mathcal{L}_{gc}$'' constraint for '+r'$\sigma_{exp}$ = ' + error + '%' + ' at ' + r'$\sqrt{s} = 13$'+' TeV\n Normalized to NNPDF3.1', fontweight='bold', fontsize=14, y=1.03)
    
    r = np.array([lumi_uw[0][1:]/lumi_uw[0][1:],lumi_rw[0][1:]/lumi_uw[0][1:]])
    e = np.array([lumi_uw[1][1:]/lumi_uw[0][1:],lumi_rw[1][1:]/lumi_uw[0][1:]])

    p = []

    for plot, ax in enumerate(axs.flatten()):
        ax2 = inset_axes(axs[plot], width = '50%', height = '25%', loc = 'lower left', bbox_to_anchor=(0.4,0.07,1.1,1.1), bbox_transform=axs[plot].transAxes)
        ax2.set_title('Error band width', fontsize = 6, pad = 2)
        for ir in range(2):
            ls_name, ls = linestyle_tuple[ir]
            p1 = ax.plot(np.NaN, np.NaN, ls = ls, color = rescolors[ir], alpha = 1)
            p2 = ax.fill(np.NaN, np.NaN, color = rescolors[ir], alpha=0.2)
            ax.plot(q_val, r[ir][plot], ls = ls, color = rescolors[ir], alpha = 1)
            ax.fill_between(q_val, r[ir][plot]+e[ir][plot], r[ir][plot]-e[ir][plot], color = rescolors[ir], alpha=0.2)
            p.append(p1)
            p.append(p2)
            ax2.plot(q_val, 2*e[ir][plot], ls = ls, color = rescolors[ir], alpha = 1)
        ax.set_xlim(2,12)
        ax2.set_xlim(2,12)
        ax.set_ylim(0.5,1.5)
        #ax2.set_xticks(fontsize = 6)
        #ax2.set_yticks(fontsize = 6)
        ax.set_xlabel('Q (GeV)', fontsize=12)
        ax.set_ylabel(r'$\mathcal{L}_{gc}/\mathcal{L}_{gc}(ref)$', fontsize=12)
        ax2.set_xlabel('Q (GeV)', fontsize=6, labelpad = -2)
        ax2.set_ylabel('90% C.L.', fontsize=6)
        ax2.tick_params(axis='both', which='both', labelsize=4)

        textstr = pdfset[plot+1]
        
        props1 = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(0.05, 0.90, textstr, transform=ax.transAxes, fontsize=10, va='bottom', bbox=props1)

        props2 = dict(boxstyle='round', facecolor='white', alpha=1)
        if round(R[plot+1],2) >= 1:
            ax.text(0.95, 0.90, 'No constraint', transform=ax.transAxes, fontsize=10, va='bottom', ha='right', bbox=props2)
        else:
            ax.text(0.95, 0.90, str(int(100*round(R[plot+1],2)))+'%'+' constraint', transform=ax.transAxes, fontsize=10, va='bottom', ha='right', bbox=props2)

    fig.legend([(p[0][0],p[1][0]),(p[2][0],p[3][0])],[legend[0],legend[1]],
                frameon=True, ncol=2, bbox_to_anchor=(0.5,0.985), loc=9, prop={'size':14})

    props3 = dict(boxstyle='round', facecolor='white', alpha=1)
    axs[0].text(0.995, 1.025, r'$\sqrt{s} = 13$'+' TeV', transform=axs[0].transAxes, fontsize=8, va='bottom', ha='right', bbox=props3)

    props4 = dict(boxstyle='round', facecolor='white', alpha=1)
    axs[0].text(0.005, 1.025, r'$\sigma_{exp}$ = ' + error + '%', transform=axs[0].transAxes, fontsize=8, va='bottom', ha='left', bbox=props4)

    axs[0].set_title(tag + ' at ' + detectors[iflav])

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

# Calls all necessary functions to generate the PDF and luminosity plots for a given ratio and error
def gen_plots(iflav,error):
        # PDF plot
    xc_uw,xc_rw = charm_pdf(iflav,error)
    R_pdf = rel_constraint_pdf(xc_uw[1],xc_rw[1])
    mkplot_charm_pdf(iflav,xc_uw,xc_rw,error,R_pdf)

    #     # gg luminosity
    # gglumi_uw,gglumi_rw = charm_gglumi(iflav,error)
    # R_gg = rel_constraint_lumi(gglumi_uw[1],gglumi_rw[1])
    # mkplot_charm_gglumi(iflav,gglumi_uw,gglumi_rw,error,R_gg)

    #     # gq luminosity
    # gqlumi_uw,gqlumi_rw = charm_gqlumi(iflav,error)
    # R_gq = rel_constraint_lumi(gqlumi_uw[1],gqlumi_rw[1])
    # mkplot_charm_gqlumi(iflav,gqlumi_uw,gqlumi_rw,error,R_gq)

    #     # gc luminosity
    # gclumi_uw,gclumi_rw = charm_gclumi(iflav,error)
    # R_gc = rel_constraint_lumi(gclumi_uw[1],gclumi_rw[1])
    # mkplot_charm_gclumi(iflav,gclumi_uw,gclumi_rw,error,R_gc)

# Calls plot generating function for all ratios
for error in errorvec:
    for iflav in range(nratio):
        gen_plots(iflav,error) 