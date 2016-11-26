import numpy as np
import matplotlib.pyplot as plt

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

def get_STV_Arduini(fHR, t_fHR, t_wnd=60):
    """
    Estimates STV as average of absolute differences between consecutive interwals in 60 seconds.
    ----------
    Parameters:
    ----------
    fHR: array
        Fetal heart rate values calculated with ZuzaDSP.
    t_fHR: array
        Timestamps for fHR calculated with ZuzaDSP, fs=0.4 Hz
    t_wnd: float
    
    Outputs:
    -------
    STV: array
    
    """
    STV=np.zeros(np.round_((t_fHR[len(t_fHR)-1]-t_fHR[0])/60))
    print len(t_fHR)
    RR_intervals=60000/fHR
    time_calc=0.0
    STV_idx=0
    start=0
  
        
    for i in range(len(t_fHR)):
        
        if time_calc<t_wnd:
            time_calc=t_fHR[i]-t_fHR[start]
    
        else:
            STV[STV_idx]= np.mean(np.abs(np.diff(RR_intervals[start:i-1])))
            print time_calc 
            print start
            print STV_idx
            time_calc=0
            start=i
            STV_idx+=1
            
    
    meanSTV=np.mean(np.abs(np.diff(RR_intervals)))
    stv=np.std(RR_intervals)
    avrg=np.mean(RR_intervals)
    means=np.nanmean(RR_intervals)
    
    return STV, meanSTV


file_name = "1511050945.npz"

t_fHR, fHR, fHR_bl, t_ref, fHR_ref = load_fHR(file_name)

plt.plot(t_fHR, fHR)
plt.plot(t_ref, fHR_ref)
plt.plot(t_fHR, fHR_bl)

print t_fHR
RR_intervals=60000/fHR
print RR_intervals
print fHR
stv=get_STV_Arduini(fHR,t_fHR, 60)
print stv
