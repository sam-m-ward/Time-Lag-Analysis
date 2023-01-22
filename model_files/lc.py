import numpy as np
import matplotlib.pyplot as pl
import pickle
import pandas as pd
import copy
from contextlib import suppress
import os

class LC:
    """
    LC Class Object (LC==Light Curve)

    Takes in measurements of time and flux (x and y) and their associated errors

    Upper limits in flux values also permitted

    Parameters
    ----------
    x,xerr,y,yerr: lists or arrays
        time, flux and their associated errors

    uplim_binary: array
        array of zeros or ones, if 1 then flux point is an upper limit

    designation,name: strs
        designation is keyword for object e.g. 'F1', name is name object e.g. '3c454.3'
    """

    def __init__(self,x,xerr,y,yerr,uplim_binary,designation,name):
        if xerr is None:
            xerr = np.zeros(len(x))
        if uplim_binary is None:
            uplim_binary = np.zeros(len(x))

        x      = np.asarray(x)
        xerr   = np.asarray(xerr)
        y      = np.asarray(y)
        yerr   = np.asarray(yerr)
        uplims = np.array(uplim_binary, dtype = bool)

        df      = pd.DataFrame(data = dict(zip(['x','xerr','y','yerr','uplim_binary','uplims'],[x,xerr[0],y,yerr,uplim_binary,uplims])))
        df.sort_values('x',ascending=True,inplace=True)

        self.df          = df
        self.designation = designation
        self.name        = name

    def remove_upper_lims(self):
        self.df = self.df[self.df['uplims']==False]

def load_data(designation,name):
    '''
    Load Data

    Simple Function to Load in Raw Data

    Parameters
    ----------
    designation, name: strs
        designation is keyword for object e.g. 'F1', name is name object e.g. '3c454.3'
    '''
    with open(f'data/Fermi{designation}.pkl','rb') as f:
        Fermi_Master = pickle.load(f)
    Ftimes,Fterrs,Fflux,Ferrs,Fuplim_binary = Fermi_Master['ts'],Fermi_Master['terrs'],Fermi_Master['fs'],Fermi_Master['ferrs'],Fermi_Master['uplim_binary']
    lcFermi = LC(Ftimes,Fterrs,Fflux,Ferrs,Fuplim_binary,designation,name)

    with open(f'data/SMA{designation}.pkl','rb') as f:
        SMA_Master = pickle.load(f)
    Stimes,Sterrs,Sflux,Serrs,Suplim_binary = SMA_Master['ts'],SMA_Master['terrs'],SMA_Master['fs'],SMA_Master['ferrs'],SMA_Master['uplim_binary']
    lcSMA = LC(Stimes,Sterrs,Sflux,Serrs,Suplim_binary,designation,name)

    return lcFermi, lcSMA

class LCpair:
    """
    LCpair class object

    Class used to analyse a pair of time-series and find correlations/time-lags using DCCFs

    Parameters
    ----------
    choices: dict
        dictionary of analysis/plotting choices

    lc1,lc2: LC class objects
        the pair of time series to jointly analyse

    designation, name: strs
        designation is keyword for object e.g. 'F1', name is name object e.g. '3c454.3'
    """
    def time_trim(self,lc1,lc2):
        xmin,xmax = lc1.df.x.min(),lc1.df.x.max()

        df2    = lc2.df
        df2    = df2[(xmin<=df2['x'])&(df2['x']<=xmax)]
        lc2.df = df2

        return lc1, lc2

    def set_LC_pair(self,lc1,lc2):
        self.lc1 = lc1
        self.lc2 = lc2
        self.LCpair = [lc1,lc2]


    def __init__(self,choices,lc1,lc2,designation,name):
        if choices['time_trim']:
            lc1,lc2 = self.time_trim(lc1,lc2)

        if choices['remove_upper_lims']:
            lc1.remove_upper_lims()
            lc2.remove_upper_lims()

        self.choices     = choices
        self.designation = designation
        self.name        = name
        self.set_LC_pair(lc1,lc2)


    def plot_lcs(self):
        '''
        Plot LCs

        Simple Plotting Function to show light curve pair
        '''

        FS    = self.choices['FS']
        fac,cc,lab = self.choices['fac'],self.choices['cc'] ,self.choices['lab']
        alpha = 0.25 ; capsize = 1 ; elinewidth = 0.5 ; marker = 'o' ; markersize = 2
        lc1   = self.lc1
        lc2   = self.lc2

        pl.figure()
        pl.title(f'{self.name} Light-Curves', fontsize=FS-3)
        for il,llc in enumerate(self.LCpair):
            lc         = llc.df
            pl.errorbar(lc.x, lc.y*fac[il], yerr = lc.yerr*fac[il], xerr = [lc.xerr],
                            uplims     = lc.uplims,
                            linestyle  = 'None',
                            marker     = marker,
                            markersize = markersize,
                            color      = cc[il],
                            ecolor     = 'black',
                            capsize    = capsize,
                            elinewidth = elinewidth
                            )
            pl.plot(lc.x, lc.y*fac[il],  linestyle = '-', linewidth=1,  color = 'black', alpha=alpha,marker=None)
            pl.scatter(lc.x.values[0],lc.y.values[0]*fac[il], marker=marker, s = 5, color = cc[il], label = lab[il])

        pl.yscale('log')
        pl.xlabel("Time (MJD)", fontsize=FS)
        pl.ylabel("Flux (Arbitrary Units)", fontsize=FS)#($MeV cm^{-2} s^{-1}$)")
        pl.legend(fontsize=FS-8, loc='upper right')
        pl.tick_params(labelsize=FS-3)
        pl.tight_layout()
        pl.show()

    def get_average_cadence(self):
        lc1,lc2 = self.LCpair[:]
        N1 = lc1.df.shape[0] ; N2 = lc2.df.shape[0]
        lower_sampled_lc = lc1 if N1<=N2 else lc2
        average_cadence  = np.average(lower_sampled_lc.df.x.values[1:]-lower_sampled_lc.df.x.values[:-1])
        self.average_cadence = average_cadence

    def get_nbins_for_DCCF(self):
        #Ensure 2/3 of LC are overlapping at any one time, as in Liodakis "Multiwavelength Cross-Correlations and Flaring Activity in Bright Blazars"
        DT = ((self.lc1.df.x.values[-1] - self.lc1.df.x.values[0])*2/3)*0.5
        taumin = -DT
        taumax = +DT
        nbins  = int(2*DT/self.average_cadence)
        self.taumin   = taumin
        self.taumax   = taumax
        self.nbins    = nbins
        self.stepsize = (self.taumax-self.taumin)/self.nbins
        self.ts_DCCF  = np.linspace(self.taumin+self.stepsize/2, self.taumax-self.stepsize/2, self.nbins)

    def calc_UCCF_DCCF(self):
        lc1 = self.lc1.df ; lc2 = self.lc2.df
        N1  = lc1.shape[0]; N2  = lc2.shape[0]

        tau    = np.zeros((N1,N2))
        binset = {_:[] for _ in range(self.nbins)} ; bin_mapper = {}
        BINS   = {x:copy.deepcopy(binset) for x in ['bar1','bar2','sig1','sig2']}
        for i in range(N1):
            for j in range(N2):
                tau[i,j] = lc2.x.values[j] - lc1.x.values[i]
                Nb       = int( np.floor( (tau[i,j]-self.taumin)/self.stepsize ) ) #del_t  = (tau[i,j] - self.taumin - Nb*self.stepsize) #bin_no = Nb + 1 #print (tau[i,j], self.taumin+Nb*self.stepsize+del_t)
                if tau[i,j]==self.taumax:#Correction for when tau[i,j]==taumax
                    Nb+=-1
                if 0<=Nb and Nb<self.nbins:
                    bin_mapper[f'{i}_{j}'] = Nb
                    BINS['bar1'][Nb].append(lc1.y.values[i])
                    BINS['bar2'][Nb].append(lc2.y.values[j])
        for b in range(self.nbins):
            BINS['sig1'][b] = np.std(BINS['bar1'][b])
            BINS['sig2'][b] = np.std(BINS['bar2'][b])
            BINS['bar1'][b] = np.average(BINS['bar1'][b])
            BINS['bar2'][b] = np.average(BINS['bar2'][b])

        UCCF      = np.zeros((N1,N2))
        UCCF_bins = {_:[] for _ in range(self.nbins)}
        for i in range(N1):
            for j in range(N2):
                with suppress(KeyError):
                    Nb = bin_mapper[f'{i}_{j}']
                    UCCF[i,j] = (lc1.y.values[i] - BINS['bar1'][Nb])*(lc2.y.values[j]-BINS['bar2'][Nb])/(BINS['sig1'][Nb]*BINS['sig2'][Nb])
                    UCCF_bins[Nb].append(UCCF[i,j])

        DCCF = np.array([np.average(UCCF_bins[b]) for b in UCCF_bins])
        tlag = self.ts_DCCF[np.argmax(np.abs(DCCF))]

        if DCCF[np.argmax(np.abs(DCCF))]>=0:
            correlation = 'correlation'
        else:
            correlation = 'anticorrelation'
        if tlag>=0:
            lagstr = 'lags'
        else:
            lagstr = 'leads'

        self.DCCF        = DCCF
        self.UCCF        = UCCF
        self.UCCF_bin    = UCCF_bins
        self.correlation = correlation
        self.lagstr      = lagstr
        self.tlag        = tlag
        self.bin_mapper  = bin_mapper
        self.BINS        = BINS

    def calc_DCCF_errs(self):
        lc1 = self.lc1.df ; lc2 = self.lc2.df
        N1  = lc1.shape[0]; N2  = lc2.shape[0]

        sigma_bar_bins = {_:{'sigma_bar1':[],'sigma_bar2':[]} for _ in range(self.nbins)}
        for i in range(N1):
            for j in range(N2):
                with suppress(KeyError):
                    Nb = self.bin_mapper[f'{i}_{j}']
                    sigma_bar_bins[Nb]['sigma_bar1'].append(lc1.yerr.values[i]**2)
                    sigma_bar_bins[Nb]['sigma_bar2'].append(lc2.yerr.values[j]**2)
        for Nb in range(self.nbins):
            sigma_bar_bins[Nb]['sigma_bar1'] = (sum(sigma_bar_bins[Nb]['sigma_bar1'])**0.5) / len(sigma_bar_bins[Nb]['sigma_bar1'])
            sigma_bar_bins[Nb]['sigma_bar2'] = (sum(sigma_bar_bins[Nb]['sigma_bar2'])**0.5) / len(sigma_bar_bins[Nb]['sigma_bar2'])


        alpha_UCCFs = {_:[] for _ in range(self.nbins)}
        for i in range(N1):
            for j in range(N2):
                with suppress(KeyError):
                    Nb = self.bin_mapper[f'{i}_{j}']

                    alpha_xixbar = (lc1.yerr.values[i]**2+sigma_bar_bins[Nb]['sigma_bar1']**2)**0.5
                    alpha_devx   = (1/self.BINS['sig1'][Nb])*alpha_xixbar
                    dev_x        = (lc1.y.values[i]-self.BINS['bar1'][Nb])/self.BINS['sig1'][Nb]

                    alpha_yiybar = (lc2.yerr.values[j]**2+sigma_bar_bins[Nb]['sigma_bar2']**2)**0.5
                    alpha_devy   = (1/self.BINS['sig2'][Nb])*alpha_yiybar
                    dev_y        = (lc2.y.values[j]-self.BINS['bar2'][Nb])/self.BINS['sig2'][Nb]

                    alpha_UCCF   = self.UCCF[i,j] * ( ( (alpha_devx/dev_x)**2 + (alpha_devy/dev_y)**2 )**0.5)
                    alpha_UCCFs[Nb].append(alpha_UCCF**2)

        self.err_DCCF = [((sum(alpha_UCCFs[_]))**0.5)/len(alpha_UCCFs[_]) for _ in range(self.nbins)]


    def compute_DCCF(self,compute_average_cadence=True,compute_errors=True):
        if compute_average_cadence:
            self.get_average_cadence()
        self.get_nbins_for_DCCF()
        self.calc_UCCF_DCCF()
        if compute_errors:
            self.calc_DCCF_errs()

    def compute_confidence_intervals(self):
        lc2 = self.lc2

        #Emmanoulopoulus 2013 LCs
        newgen_LCs_E2013 = np.load(f'products/E13synthLCs/{self.name}E2013_synthLCs10k.npy')
        #newgen_LCs_E2013 = newgen_LCs_E2013[:100]

        new_choices = self.choices
        new_choices['time_trim']         = False
        new_choices['remove_upper_lims'] = False

        colnames    = [f'{_}' for _ in range(self.nbins)]

        try:
            with open(f'products/{self.name}confidencecurves.pkl','rb') as f:
                confidence_curves = pickle.load(f)

        except:

            try:
                with open(f'products/{self.name}synthDCCFdftot.pkl','rb') as f:
                    dftot = pickle.load(f)
            except:
                for _ in range(len(newgen_LCs_E2013)):
                    if not os.path.exists(f'products/dfnews/{self.name}synthDCCFdftot{_}.pkl'):
                        break
                still_to_do_indices = np.arange(_,len(newgen_LCs_E2013))
                for _ in still_to_do_indices:#len(newgen_LCs_E2013)):
                    if (_+1)%10==0:
                        print (f'Computing DCCFs for Synthesised LCs of {self.name}: {_+1}/{len(newgen_LCs_E2013)}')


                    synthLC     = abs(newgen_LCs_E2013[_][0])
                    synthLCtime = abs(newgen_LCs_E2013[_][1])
                    lc1_new     = LC(synthLCtime,None,synthLC,np.zeros(len(synthLC)),None,self.designation,'synth')
                    lcpair_new  = LCpair(new_choices,lc1_new,lc2,self.designation,'synth')#Correlate new Gamma-ray LC with Radio LC
                    lcpair_new.average_cadence = self.average_cadence
                    lcpair_new.compute_DCCF(compute_average_cadence=False,compute_errors=False)

                    dfnew = pd.DataFrame(data=dict(zip(colnames,[[xi] for xi in lcpair_new.DCCF])))
                    with open(f'products/dfnews/{self.name}synthDCCFdftot{_}.pkl','wb') as f:
                        pickle.dump(dfnew,f)

                print ('Completed Synthesising')
                print ('###'*10)
                dftot = []
                for _ in range(len(newgen_LCs_E2013)):
                    with open(f'products/dfnews/{self.name}synthDCCFdftot{_}.pkl','rb') as f:
                        dfnew = pickle.load(f)
                    dftot.append(dfnew)
                dftot = pd.concat(dftot,axis=0)

                with open(f'products/{self.name}synthDCCFdftot.pkl','wb') as f:
                    pickle.dump(dftot,f)

            sigma_levels = [0.68,0.95,0.997]
            sigma_levels = np.concatenate((0.5-np.asarray(sigma_levels[::-1])/2,np.asarray(sigma_levels)/2+0.5))
            indices      = [int(s*(dftot.shape[0]-1)) for s in sigma_levels]
            mapper       = ['-3','-2','-1','1','2','3']


            confidence_curves = {mapper[_]:[] for _ in range(len(indices))}
            print ('Beginning sorting confidence intervals')
            for col in colnames:
                points = list(dftot[col].values)
                points.sort()
                for _,i in enumerate(indices):
                    confidence_curves[mapper[_]].append(points[i])
            print ('Finished')
            print ('###'*10)

        with open(f'products/{self.name}confidencecurves.pkl','wb') as f:
            pickle.dump(confidence_curves,f)

        self.confidence_curves = confidence_curves


    def plot_DCCF(self):

        pl.figure()
        #'''
        cs = ['blue','green','red']
        for ic,cc in enumerate(cs):
            upper_key = f'{ic+1}' ; lower_key = f'-{ic+1}'
            pl.plot(self.ts_DCCF, self.confidence_curves[upper_key],c=cc,label=f'{ic+1}'+r'$\sigma$')
            pl.plot(self.ts_DCCF, self.confidence_curves[lower_key],c=cc)

        #'''
        FS = self.choices['FS']
        pl.title(f"{self.name}; Radio {self.lagstr} by {abs(round(self.tlag,2))} days", fontsize=FS-3)
        pl.plot([self.tlag,self.tlag],[-1,1],c='r')
        pl.step(np.concatenate((np.array([self.taumin-self.stepsize/2]),self.ts_DCCF))+self.stepsize/2, np.concatenate((np.array([self.DCCF[0]]),self.DCCF)), c='black')
        pl.errorbar(self.ts_DCCF, self.DCCF, yerr=self.err_DCCF,
                            linestyle  = 'None',
                            marker     = 'None',
                            ecolor     = 'black',
                            capsize    = 1,
                            elinewidth = 0.5)
        pl.xlabel(r"$\tau$ / days", fontsize=FS)
        pl.ylabel('DCCF', fontsize=FS)
        pl.legend(fontsize=FS-3)
        pl.tick_params(labelsize=FS-3)
        pl.tight_layout()
        pl.show()
