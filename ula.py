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
    


file_name = "1511050945.npz"

t_fHR, fHR, fHR_bl, t_ref, fHR_ref = load_fHR(file_name)

plt.plot(t_fHR, fHR)
plt.plot(t_ref, fHR_ref)
plt.plot(t_fHR, fHR_bl)
