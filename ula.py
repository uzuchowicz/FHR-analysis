import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy


def load_fHR(file_name):
    """
    Parameters:
    -----------
    file_name : str
        File containing fHR data.
        
    Outputs:
    --------
    t_fHR : array
        Timestamps for fHR calculated with ZuzaDSP.
    fHR : array
        Fetal heart rate values calculated with ZuzaDSP.
    fHR_bl : array
        Fetal heart rate baseline.
    t_ref : array
        Timestamps for reference FHR.
    fHR_ref : array
        Reference fetal heart rate signal.
    """
    
    data = np.load(file_name)
    t_fHR = data["t_fHR"]
    fHR = data["fHR"]
    fHR_bl = data["fHR_bl"]
    t_ref = data["t_ref"]
    fHR_ref = data["fHR_ref"]
    
    return t_fHR, fHR, fHR_bl, t_ref, fHR_ref
 
    ###################BASAL HEART RATE #########################################
    
def get_Basal_fHR(fHR, t_fHR, app_val = 5.0, t_wnd = 30, max_amp = 10, max_val = 47.5, min_val = 222.5):
    """
    Estimate basal fetal heart rate value in stable segments fHR for evaluates potential arrythmias.
    --------
    Parameters:
    --------
    fHR: array
        Fetal heart rate values calculated with ZuzaDSP.
    t_fHR: array
        Timestamps for fHR calculated with ZuzaDSP, fs=0.4 Hz
    app_val: int
        Baseline is rounded to app_val (bpm).
    t_wnd: float
        Time window for detection stable segments in seconds.
    max_amp: float
        Maximum amplitude of stable segments in bpm.
    min_val: float
        Minimum value of fHR to estimate basal fHR.
    max_val: float
        Maximum value of fHR to estimate basal fHR.
    Outputs:
    ---------
    Basal_FHR: array
        Basal fetal heart rate value.
    fHR_stable: array 
        Stable FHR segments. Unstable parts replaced with NaNs. 
    prc_stable_fHR: float
        Percent of stable fHR in record.
    Bradycardia: bool
        True if detected basal fHR below normal values.
    Tachycardia: bool
        True if detected basal fHR more than normal values.
    """
    fHR = copy.deepcopy(fHR)
    
    Ts = np.mean(np.diff(t_fHR)) 
    fHR_stable = fHR
    n_samp = int(t_wnd / Ts)
    N=len(fHR)
     
    for i in range(N - n_samp):
        fHR_wnd = fHR[i : i + n_samp]
        if (~np.isnan(fHR_wnd)).all():
            amp_wnd = np.nanmax(fHR_wnd) - np.nanmin(fHR_wnd)
            if amp_wnd > max_amp:
                fHR_stable[i : i + n_samp] = np.nan  
             
    fHR_stable_sgmts = np.delete(fHR_stable, np.nonzero(np.isnan(fHR_stable)))
    prc_stable_fHR = float(len(fHR_stable_sgmts))/float(len(fHR))*100  

    bins = np.arange(max_val, min_val, app_val) 
    hist_fHR = np.histogram(fHR_stable_sgmts, bins)
    basal_fHR = bins[np.nanargmax(hist_fHR[0])] + (app_val*0.5) 
    basal_fHR = int(basal_fHR)
    #plt.hist(fHR_stable_sgmts, bins)
    #plt.show()
   
    Bradycardia = False
    Tachycardia = False
    if basal_fHR>150:
        Tachycardia = True
    if basal_fHR<110:
        Bradycardia = True
       
    return basal_fHR, fHR_stable, prc_stable_fHR, Bradycardia, Tachycardia
    
#################SHORT-TIME VARIABILITY ###########################    

def get_STV_Arduini(fHR, t_fHR, t_wnd = 60):
    """
    Estimate STV as average of absolute differences between consecutive interwals in 60 seconds.
    ----------
    Parameters:
    ----------
    fHR: array
        Fetal heart rate values calculated with ZuzaDSP.
    t_fHR: array
        Timestamps for fHR calculated with ZuzaDSP, fs=0.4 Hz
    t_wnd: float
        Time window in second.
    Outputs:
    -------
    STV_wnd: array
        Short-term variability values in every time window.
    STV: float
        Short-term variability value for fHR record.
    
    """
    Ts = np.mean(np.diff(t_fHR)) 
    RR_intervals = 60000/fHR
    n_samp = int(t_wnd/Ts)
    STV_wnd = np.zeros(int(len(fHR)/n_samp))
    
    for i in range(len(STV_wnd)):
        RR_wnd=RR_intervals[i*n_samp : (i+1)*n_samp]
        STV_wnd[i] = np.mean(np.abs(np.diff(RR_wnd)))
    
    STV = np.nanmean(np.abs(np.diff(RR_intervals)))
    
    return STV_wnd, STV

def get_STV_Haan(fHR, t_fHR, n_intervals=128):
    """
    Estimate STV as a interquartile range of angular condidates of points corresponding to consecutive pairs of
    intervals .
    ----------
    Parameters:
    ----------
    fHR: array
        Fetal heart rate values calculated with ZuzaDSP.
    t_fHR: array
        Timestamps for fHR calculated with ZuzaDSP, fs=0.4 Hz
    n_intervals
        Number of real intervals to estimate de Haan idex.
    
    Outputs:
    -------
    STV_wnd: array
        Short-term variability indexes in time of n_intervals.
    STV: float
        Short-term variability index for fHR record.
     """   
    Ts = np.mean(np.diff(t_fHR))
    RR_intervals = 60000/fHR
    
    n_samp = int(n_intervals * np.nanmean(RR_intervals) / (Ts*1000))
    n_wnd=int(np.round_(len(fHR)/n_samp))
    STV_wnd = np.zeros(n_wnd)
    RR_points = np.zeros((n_samp-1,2))
    
    RR_phi_wnd = np.zeros(n_samp-1) #gdzie umiescić tę zmienną? Czy mogę nie deklarowac zmiennych RR_points? Tylko w samej pętli?
    for i in range(n_wnd):
        for j in range(n_samp-1):
            RR_points[j,0] = RR_intervals[i*n_samp+j]
            RR_points[j,1] = RR_intervals[i*n_samp+j+1]
            RR_phi_wnd[j] = np.arctan(RR_points[j,1]/RR_points[j,0])
            
        STV_wnd[i] = np.subtract(*np.nanpercentile(RR_phi_wnd, [75, 25]))
        
    RR_phi = np.zeros(len(RR_intervals)-1) 
    RR_points = np.zeros((len(RR_intervals)-1,2))
    for i in range(len(RR_intervals) - 1):
         RR_points[i,0] = RR_intervals[i]
         RR_points[i,1] = RR_intervals[i+1]
         RR_phi[i] = np.arctan(RR_points[i,1]/RR_points[i,0])
         
    STV = np.subtract(*np.nanpercentile(RR_phi,[75, 25]))
    
    return STV_wnd, STV

def get_STV_Yeh(fHR, t_fHR, t_wnd=60):
    """
    Estimate STV as standard deviation of D indexes. D index is a ratio between difeerences and sum of consecutive pairs of intervals. 
    ----------
    Parameters:
    ----------
    fHR: array
        Fetal heart rate values calculated with ZuzaDSP.
    t_fHR: array
        Timestamps for fHR calculated with ZuzaDSP, fs=0.4 Hz
    t_wnd: float
        Time window.
    
    Outputs:
    -------
    STV_wnd: array
        Short-term variabilite indexes in consecutive time windows.
    STV_wnd: float
        Short-term variabilite indexes fHR record.
     """   
    Ts = np.mean(np.diff(t_fHR))
    RR_intervals = 60000/fHR
    n_samp = int(t_wnd/Ts)
    N=len(fHR)
    STV_wnd = np.zeros(int(N/n_samp)) 
    
    D_ind = np.zeros(n_samp-1)
    
    for i in range(len(STV_wnd)):
        for j in range(n_samp-1):
            RR_1 = RR_intervals[n_samp*i+j]
            RR_2 = RR_intervals[n_samp*i+j+1]
            D_ind[j] = (RR_2 - RR_1)/(RR_1 + RR_2)*1000
        STV_wnd[i] = np.nanstd(D_ind)
        
    D_ind = np.zeros(N-1)
    for i in range(N-1):
        RR_1 = RR_intervals[i]
        RR_2 = RR_intervals[i+1]
        D_ind[i] = (RR_2 - RR_1)/(RR_1+RR_2)*1000
    STV = np.nanstd(D_ind)   
        
    return STV_wnd, STV
    
def get_STV_Huey(fHR, t_fHR, t_wnd=30):
    """
    Estimate STV as sum of absolute differences between two consecutive intervals providing of changing monotonicity.
    ----------
    Parameters:
    ----------
    fHR: array
        Fetal heart rate values calculated with ZuzaDSP.
    t_fHR: array
        Timestamps for fHR calculated with ZuzaDSP, fs=0.4 Hz
    t_wnd: float
        Time window.
    
    Outputs:
    -------
    STV_wnd: array
        Short-term variabilite indexes in consecutive time windows.
    STV_wnd: float
        Short-term variabilite indexes fHR record.
     """   
    Ts = np.mean(np.diff(t_fHR))
    RR_intervals = 60000/fHR
    n_samp = int(t_wnd/Ts)
    N=len(fHR)
    STV_wnd = np.zeros(int(N/n_samp)) 
    
    RR_diff = np.diff(RR_intervals)
       
    for i in range(len(STV_wnd)):
        for j in range(n_samp-2):
            RR_diff_1 = RR_diff[n_samp*i + j]
            RR_diff_2 = RR_diff[n_samp*i + j+1]
            if RR_diff_1*RR_diff_2 < 0:
                STV_wnd[i] = STV_wnd[i] + np.abs(RR_diff_2)
        
    STV = 0
    for i in range(N-2):
        RR_diff_1 = RR_diff[i]
        RR_diff_2 = RR_diff[i + 1]
        if RR_diff_1*RR_diff_2 < 0:
            STV = STV + np.abs(RR_diff_2)
    STV = STV/int(len(STV_wnd))
        
    return STV_wnd, STV

def get_STV_Dalton(fHR, t_fHR, t_wnd=60):
    """
    Estimate STV as half mean of absolute differences between two consecutive intervals in time window.
    ----------
    Parameters:
    ----------
    fHR: array
        Fetal heart rate values calculated with ZuzaDSP.
    t_fHR: array
        Timestamps for fHR calculated with ZuzaDSP, fs=0.4 Hz
    t_wnd: float
        Time window.
    
    Outputs:
    -------
    STV_wnd: array
        Short-term variabilite indexes in consecutive time windows.
    STV_wnd: float
        Short-term variabilite indexes fHR record.
     """   
    Ts = np.mean(np.diff(t_fHR))
    RR_intervals = 60000/fHR
    n_samp = int(t_wnd/Ts)
    N=len(fHR)
    STV_wnd = np.zeros(int(N/n_samp))   
    RR_abs_diff = np.abs(np.diff(RR_intervals))
      
    for i in range(len(STV_wnd)):
            RR_abs_diff_wnd = RR_abs_diff[i*n_samp : (i+1)*n_samp]
            STV_wnd[i] = 0.5 * np.abs(np.nanmean(RR_abs_diff_wnd))
        
    STV =  0.5 * np.abs(np.nanmean(RR_abs_diff))
     
    return STV_wnd, STV       
    
def get_STV_van_Geijn(fHR, t_fHR, t_wnd=30):
    """
    Estimate STV as interquartile range of weighted differences between two consecutive intervals in time window.
    ----------
    Parameters:
    ----------
    fHR: array
        Fetal heart rate values calculated with ZuzaDSP.
    t_fHR: array
        Timestamps for fHR calculated with ZuzaDSP, fs=0.4 Hz
    t_wnd: float
        Time window.
    
    Outputs:
    -------
    STV_wnd: array
        Short-term variabilite indexes in consecutive time windows.
    STV_wnd: float
        Short-term variabilite indexes fHR record.
     """   
    Ts = np.mean(np.diff(t_fHR))
    RR_intervals = 60000/fHR
    n_samp = int(t_wnd/Ts)
    N=len(fHR)
    STV_wnd = np.zeros(int(N/n_samp)) 
       
    for i in range(len(STV_wnd)):
        RR_param = np.zeros(n_samp)
        for j in range(n_samp - 1):
            RRs = RR_intervals[n_samp*i + j : n_samp*i + j+2]            
            RR_mean = np.mean(RRs)
            g = np.power(180/(RR_mean - 320), 1.5)
            RR_param[j] = g * (np.abs(np.diff(RRs)))
        STV_wnd[i] = np.subtract(*np.nanpercentile(RR_param,[75, 25]))
        
    RR_param = np.zeros(N)
    for i in range(N - 1):    
            RRs = RR_intervals[i : i+2]            
            RR_mean = np.mean(RRs)
            g = np.power(180/(RR_mean - 320), 1.5)
            RR_param[i] = g * (np.abs(np.diff(RRs)))
            
    STV = np.subtract(*np.nanpercentile(RR_param,[75, 25]))
    
    return STV_wnd, STV          
######################LONG-TERM VARIABILITY##############################    

def get_Oscillation_Index(fHR, t_fHR, t_wnd=600 ):
    """
    Estimate LTV as average of absolute differences between consecutive interwals in time window (default 60 seconds).
    ----------
    Parameters:
    ----------
    fHR: array
        Fetal heart rate values calculated with ZuzaDSP.
    t_fHR: array
        Timestamps for fHR calculated with ZuzaDSP, fs=0.4 Hz
    t_wnd: float
        Time window 
    Outputs:
    -------
    LTV: array
        Long-term variability values as Oscillation Index in every time windows.
    
    """
    Ts = np.mean(np.diff(t_fHR))
    RR_intervals = 60000/fHR
    n_samp = int(t_wnd/Ts)
    LTV = np.zeros(int(len(fHR)/n_samp))
    
    for i in range(len(LTV)):
        RR_wnd = RR_intervals[i*n_samp : (i+1)*n_samp]
        LTV[i] = np.abs(np.nanmax(RR_wnd) - np.nanmin(RR_wnd))
           
    return LTV

def get_LTV_Haan(fHR, t_fHR, n_intervals=128):
    """
    Estimates LTV as interquartile range of modulus condidates of points corresponding to consecutive pairs of
    intervals .
    ----------
    Parameters:
    ----------
    fHR: array
        Fetal heart rate values calculated with ZuzaDSP.
    t_fHR: array
        Timestamps for fHR calculated with ZuzaDSP, fs=0.4 Hz
    n_intervals: float
        Number of real intervals to estimate de Haan LTV index.
    
    Outputs:
    -------
    LTV_wnd: array
        Long-term variability indexes in time of consecutive n_intervals.
    LTV:
        Long-term variability index in fHR record.
     """   
    Ts = np.mean(np.diff(t_fHR))
    RR_intervals = 60000/fHR
    
    n_samp = int(n_intervals * np.nanmean(RR_intervals) / (Ts*1000))
    n_wnd = int(len(fHR) / n_samp)
    LTV_wnd = np.zeros(n_wnd)
    RR_points = np.zeros((n_samp-1, 2))
    RR_modulus = np.zeros(n_samp-1)
    
    for i in range(len(LTV_wnd)):
        for j in range(n_samp-1):
            RR_points[j,0] = RR_intervals[i*n_samp+j]
            RR_points[j,1] = RR_intervals[i*n_samp+j+1]
            RR_modulus[j] = np.sqrt(np.power(RR_points[j,1], 2) + np.power(RR_points[j,0], 2))
            
        LTV_wnd[i] = np.subtract(*np.nanpercentile(RR_modulus, [75, 25]))
   
    RR_modulus = np.zeros(len(RR_intervals)-1) 
    RR_points = np.zeros((len(RR_intervals)-1, 2))
    
    for i in range(len(RR_intervals) - 1):
         RR_points[i,0] = RR_intervals[i]
         RR_points[i,1] = RR_intervals[i+1]
         RR_modulus[i] = np.sqrt(np.power(RR_points[i,1],2) + np.power(RR_points[i,0],2))
         
    LTV = np.subtract(*np.nanpercentile(RR_modulus,[75, 25]))
    
    return LTV_wnd, LTV
 
def get_LTV_Yeh(fHR, t_fHR, t_wnd=60):
    """
    Estimate LTV as a ratio between standard deviation of intervals RR and their mean in specified time window. 
    ----------
    Parameters:
    ----------
    fHR: array
        Fetal heart rate values calculated with ZuzaDSP.
    t_fHR: array
        Timestamps for fHR calculated with ZuzaDSP, fs=0.4 Hz
    t_wnd: float
        Time window.
    
    Outputs:
    -------
    LTV: array
        Long-term variabilite indexes in consecutive time windows.
     LTV:
        Long-term variability index in fHR record.  
     """   
    Ts = np.mean(np.diff(t_fHR))
    RR_intervals = 60000/fHR
    n_samp = int(t_wnd/Ts)
    LTV_wnd = np.zeros(int(len(fHR)/n_samp))
    
    for i in range(len(LTV_wnd)):
        RR_wnd=RR_intervals[n_samp*i : (i+1)*n_samp]
        LTV_wnd[i] = np.nanstd(RR_wnd) / np.nanmean(RR_wnd)
    
    LTV = np.nanstd(RR_intervals) / np.nanmean(RR_intervals) 
    
    return LTV_wnd, LTV 
    
def get_LTV_Huey(fHR, t_fHR, t_wnd=60):
    """
    Estimate LTV as sum of absolute differences between two consecutive intervals providing of longer changing monotonicity.
    ----------
    Parameters:
    ----------
    fHR: array
        Fetal heart rate values calculated with ZuzaDSP.
    t_fHR: array
        Timestamps for fHR calculated with ZuzaDSP, fs=0.4 Hz
    t_wnd: float
        Time window.
    
    Outputs:
    -------
    LTV_wnd: array
        Short-term variabilite indexes in consecutive time windows.
    LTV_wnd: float
        Short-term variabilite indexes fHR record.
     """   
    Ts = np.mean(np.diff(t_fHR))
    RR_intervals = 60000/fHR
    n_samp = int(t_wnd/Ts)
    N=len(fHR)
    LTV_wnd = np.zeros(int(N/n_samp))   
    RR_diff = np.diff(RR_intervals)
       
    for i in range(len(LTV_wnd)):
        for j in range(n_samp-3):
            RR_diff_1 = RR_diff[n_samp*i + j]
            RR_diff_2 = RR_diff[n_samp*i + j+1]
            RR_diff_3 = RR_diff[n_samp*i + j+2]
            if RR_diff_1*RR_diff_2*RR_diff_3 > 0:
                LTV_wnd[i] = LTV_wnd[i] + np.abs(RR_diff_2)
        
    LTV = 0
    for i in range(N-3):
        RR_diff_1 = RR_diff[i]
        RR_diff_2 = RR_diff[i + 1]
        RR_diff_3 = RR_diff[i + 2]
        if RR_diff_1 * RR_diff_2 * RR_diff_3 > 0:
            LTV = LTV + np.abs(RR_diff_2)
     
    LTV = LTV/float(len(LTV_wnd))    
        
    return LTV_wnd, LTV
          
####################ENTROPY##################################
  
def get_ApEn(fHR,t_fHR, m=2, r_mlp=0.5, wnd = False, t_wnd = 60): 
    """
    Estimate Approximate Entropy in fHR record.
    ----------
    Parameters:
    ----------
    fHR: array
        Fetal heart rate values calculated with ZuzaDSP.
    t_fHR: array
        Timestamps for fHR calculated with ZuzaDSP, fs=0.4 Hz
    m: int
        Lenght of compared RR intervals sequences.
    r_mlp: float
        Multiple standard deviation. Tolerance for accepting matches is standard deviation multiplied by r_mpl.
    
    Outputs:
    -------
    LTV: array
        Long-term variabilite indexes in consecutive time windows.
     LTV:
        Long-term variability index in fHR record.
    """
    RR_intervals = 60000/fHR  
    r = r_mlp*np.nanstd(RR_intervals)
    N = len(RR_intervals)  
    Phi_m_r = np.zeros(2)
    for n in range(2):
        m = m+n
        Pm = np.zeros((N-m+1, m))
       
        for j in range(N - m+1):
            for i in range(m):
                Pm[j, i] = RR_intervals[j+i]
        
        pm_distances = np.zeros((N-m+1,N-m+1))
        for i in range(N-m+1):
            for j in range(N-m+1):  
                dist = np.zeros(m)
                for k in range(m): 
                    dist[k] = np.abs(Pm[j,k]-Pm[i,k])
                    pm_distances[i,j] = np.nanmax(dist) 
                    pm_distances[j,i] = np.nanmax(dist)
                    
        
        pm_similarity = pm_distances>r 
        
        C_m_r = np.zeros(N-m+1)
        for i in range(N-m+1):
            n_i = np.nansum(pm_similarity[i])
            C_m_r[i] = float(n_i) / float(N)
        
        Phi_m_r[n] = np.nanmean(np.log(C_m_r))
    ApEn = np.abs(Phi_m_r[0] - Phi_m_r[1])
    
        
    return ApEn
    

####################ACCELERATIONS############################

def fHR_acc_det(fHR, t_fHR, min_amp=10, min_duration=30, max_duration=120, max_t_incr=30):
    """
    Detect accelerations in fHR record based on 3 criterias.
    ----------
    Parameters:
    ----------
    fHR: array
        Fetal heart rate values calculated with ZuzaDSP.
    t_fHR: array
        Timestamps for fHR calculated with ZuzaDSP, fs=0.4 Hz
    min_amp: float
        Minimum amplitude of detected acceleration.
    min_duration: float
        Minimum acceleration duration (oneset to recovery)
    min_t_incr: float
        Minimum time of acceleration increasing (onset to peak)
    Outputs:
    -------
    FHR_acc: array
        fHR during detected accelerations. Other parts replaced with NaNs.
    """
    fHR_acc = copy.deepcopy(fHR)
    
    Ts = np.mean(np.diff(t_fHR))
    N = len(fHR)
    is_acc = np.zeros(N) 
    
    basal_fHR, fHR_stable, prc_stable_fHR, Bradycardia, Tachycardia = get_Basal_fHR(fHR,t_fHR)
   
    
    fHR_diff = np.diff(fHR)
    n_acc = 0
    n_max_samp = int(max_duration/Ts)
    n_min_samp = int(min_duration/Ts)
    end_acc = 0
    criteria = np.zeros(3) 
    
    for i in range(int(N - n_max_samp)):
        if i>end_acc and fHR[i]>basal_fHR and fHR_diff[i]>0 and fHR_diff[i+1]>0 and fHR_diff[i+2]>0 and fHR_diff[i+3]>0:
            
            for j in range(n_max_samp):
                if fHR[i+j] < basal_fHR:
                    
                    fHR_wnd = fHR[i: i+j]
                   
                    amp = np.nanargmax(fHR_wnd)
                    
                    if j > n_min_samp:
                        criteria[0]=1

                    if fHR_wnd[amp] > min_amp:
                        criteria[1]=1   
 
                    if amp < np.round_(max_t_incr/Ts):
                        criteria[2]=1

                    if criteria.all() :
                      
                        is_acc[i:i+j] = 1
                        n_acc += 1
                    end_acc=i+j
                      
                    break
       
         
            
    fHR_acc = np.where(is_acc == 0, np.nan, fHR_acc)
            
    return fHR_acc
    
def get_acc_param(fHR, fHR_acc, t_fHR):
    """
    Calculate acceleration parameters - lenghts and areas - based on detected acceleration segments of fHR.
    ----------
    Parameters:
    ----------
    fHR: array
        Fetal heart rate values calculated with ZuzaDSP.
    fHR_acc: array
        Fetal heart rate values during detected accelerations.
    t_fHR: array
        Timestamps for fHR calculated with ZuzaDSP, fs=0.4 Hz.
        
    Outputs:
    -------
    acc_lenght: float
        Duration of every detected accelerations in seconds.
    acc_area: float
        Area of every detected accelerations in seconds.
    acc_amp: float
        Amplitude (above basal fHR) of every detected accelerations.
    n_acc: int
        Number of detected accelerations.

    
    
    
"""
    fHR = copy.deepcopy(fHR)
    fHR_acc = copy.deepcopy(fHR_acc)
    
    acc_lenght = []
    acc_area = []
    acc_amp = []
    n_acc=0
    
    Ts=np.mean(np.diff(t_fHR))
    N=len(fHR_acc)
    basal_fHR, fHR_stable, prc_stable_fHR, Bradycardia, Tachycardia  = get_Basal_fHR(fHR, t_fHR)  
    
    end_acc = 0
    for i in range(N): 
        if i > end_acc and ~np.isnan(fHR_acc[i]):
            for j in range(N-i):
                if np.isnan(fHR_acc[i+j+1]):
                            
                            acc_lenght.append(Ts*j)
                            
                            acc = fHR_acc[i : i+j+1] 
                            area = np.trapz(acc, dx=Ts) - basal_fHR*Ts*(j) 
                            acc_area.append(area) 
                            amp = np.nanmax(acc) - basal_fHR 
                            acc_amp.append(amp)  
                            n_acc += 1                           
                            end_acc = i+j
                            
                            break
    return acc_lenght, acc_area, acc_amp, n_acc
    
 #################POWER SPECTRUM############################
"""
def get_pow_spectr(fHR,t_fHR, LF=0.03-0.15, MF=0.15-0.5,HF=0.5-2):
    
   return 
"""             
##################SINUSOIDAL RHYTHM#################################

def sin_rhythm(LTV_function, STV_function, fHR, t_fHR, wnd, min_ratio, SD=False):
    """
    Evaluate occurence of sinusoidal rhythm based on LTV/STV ratios.
    ----------
    Parameters:
    ----------
    LTV_function: function
        Function to estimate long-term variability values in window.
    STV_function: function
        Function to estimate long-term variability values in window.
    fHR: array
        Fetal heart rate values calculated with ZuzaDSP.
    t_fHR: array
        Timestamps for fHR calculated with ZuzaDSP, fs=0.4 Hz
    min_ratio: float
        Threshold for LTV/STV ratio for sinusoidal rhythm.
    wnd: int
        Window in second or intervals.
        
    
    Outputs:
    -------
    fHR_sin: array
        fHR values in potentialy sinusoidal rhythm.
    
    """
    fHR_sin=copy.deepcopy(fHR)
    
    LTV, aLTV = LTV_function(fHR,t_fHR,wnd)
    STV, aSTV = STV_function(fHR,t_fHR, wnd)
    
    is_sin_fHR = np.zeros(len(t_fHR))
    Ts = np.nanmean(np.diff(t_fHR))
    
    n_samp=int(wnd/Ts)
    ratios=np.zeros(len(STV))
    
    if SD==True:
        ratios = [ltv / stv for ltv, stv in zip(LTV, STV)]
        
        min_ratio = np.nanmean(ratios) + np.nanstd(ratios)
        
   
    for i in range(len(STV)):
      ratio = LTV[i]/STV[i]
    
      
      if ratio > min_ratio: 
          is_sin_fHR[i*n_samp:(i+1)*n_samp]=True
      else:
          is_sin_fHR[i*n_samp:(i+1)*n_samp]=False
                       
    for i in range(len(fHR)):
        if not is_sin_fHR[i]: 
            fHR_sin[i]=np.nan
       
    return fHR_sin
        
#######################################################
def fHR_windows(fHR, t_fHR, wnd, time = True):
    Ts = np.mean(np.diff(t_fHR))
    bnd_wnd_fHR = np.zeros(len(fHR))
    if time:
        n_samp = int(wnd/Ts)
        n_wnd = int(len(fHR)/n_samp)
        for i in range(n_wnd):
            bnd_wnd_fHR[n_samp*i] = fHR[n_samp*i]
        bnd_wnd_fHR = np.where(bnd_wnd_fHR == 0, np.nan, bnd_wnd_fHR)
        
    if not time:
        Ts = np.mean(np.diff(t_fHR))
        RR_intervals = 60000/fHR   
        n_samp = int(wnd * np.nanmean(RR_intervals) / (Ts*1000))
        n_wnd = int(len(fHR) / n_samp)
        for i in range(n_wnd):
            bnd_wnd_fHR[n_samp*i] = fHR[n_samp*i]
        bnd_wnd_fHR = np.where(bnd_wnd_fHR == 0, np.nan, bnd_wnd_fHR)
     
    return bnd_wnd_fHR
    
######################################################
file_name = "1511050945.npz"
file_name2="1502021138.npz"
file_name3="1606291005.npz"

t_fHR, fHR, fHR_bl, t_ref, fHR_ref = load_fHR(file_name2)

fHR_ref = [val[0] for val in fHR_ref]
t_ref = [val[0] for val in t_ref]
t_ref=np.asarray(t_ref)
fHR_ref = np.asarray(fHR_ref)


x=fHR_windows(fHR, t_fHR, 60, True)


########################################################





########################################################
plt.plot(t_fHR, x, 'ro')
plt.plot(t_fHR, fHR_bl)
plt.plot(t_fHR, fHR)
plt.savefig("fHR.png", dpi=1000)







