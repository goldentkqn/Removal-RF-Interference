import pandas as pd
import numpy as np
import pywt

#=======================================================================================================================================#
# This algorithm used to prepair for American Society for Mass Spectrometry written by Đặng Công Minh on 30/12/2022,                    #
# at Biophysics Mass Spectrometry Lab, Department of Physics, National Dong Hwa University, Shoufeng, Hualien, Taiwan.                  #
#=======================================================================================================================================#

def Estimate_threshold(data,wavelet,level):
    r"""
    This is function to identify noise components;
    data is input your data;
    wavelet is which wavelet your choice to your data;
    levels is highest of decomposition level your choice.
    The result of this function will be used to evalute "n" parameter in Denoising_using_wavelet_base_method()

    """
    Dec  = pywt.wavedec(data,wavelet,level=level) # Decomposition raw signal;
    def energy(data):
        s = 0
        for i in data:
            s += i**2
        return s
    Mean = [np.mean(i) for i in Dec]
    Median = [np.median(i) for i in Dec]
    Energy = [energy(i) for i in Dec]
    
    return Mean, Median, Energy

#========================================================================================================================================#

def Denoising_using_wavelet_base_method(data,wavelet,level,j,n):
    r"""
    This is function to wavelet packet decomposition;
    data is input your data;
    wavelet is which wavelet your choice to your data;
    levels is highest of decomposition level your choice;
    j to esimate a parameter computed by median of coefficent detail in each level;
    n is estimate levels you want to denoising.

    """
    coeff  = pywt.wavedec(data,wavelet,level=level) # Decomposition raw signal;
    Dec  = pywt.wavedec(data,wavelet,level=level) # Decomposition raw signal;
    sigma = (1/0.6745)*np.median(np.abs(coeff[j])) # Esimate sigma coefficent;
    uthresh = sigma * np.sqrt(2 * (np.log(len(data)))/np.log(np.e)) # Estimate a hard threshold based on sigma coefficent;
    coeff[n:] = (pywt.threshold(i, uthresh, mode='hard') for i in coeff[n:])# Denoising sinusoidal interference and reducing white noise;
    rec = pywt.waverec(coeff,wavelet,'symmetric') # Convolve cDn and cAn -> Reconstruction.
    len_dec = len(Dec)
    for i in range(0,len_dec): # -> Extract raw signal become cAn, cDn, cDn-1,..., cD1.
        try:
            dec = Dec[i] # Decomposition level without sinusoidal interference!
            coe = coeff[i] # Decomposition level with noise!
        except:
            pass
        #...
        #...
        #...
        #plot your wavelet decomposition data here!
    return rec

#=======================================================================================================================================#

def wavelet_selection(raw,levels):
    r"""
    This is function to select wavelet based on rate energy-to-Shannon Entropy;
    raw is input your data;
    levels is highest of decomposition level your choice.

    """
    arrJ = []
    arrJ_Etot = []
    arrJ_S = []

    def shannon(dec_data):
        Si = []
        E=[np.abs(i)**2 for i in dec_data]
        Etot = np.sum(E)
        for i in dec_data:
            E=np.abs(i)**2
            P=E/Etot
            #Entropy = -P*(np.log2(P, where = P > 0))
            Entropy = -P*(np.log2(P))
            Si.append(Entropy)
        Stot = np.sum(Si)
        try:
            ratio=Etot/Stot
        except ZeroDivisionError:
            ratio = 0
        return ratio, Etot, Stot, E
    
    wavelet = ['haar', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14','db15', 
    'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28',
    'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38','bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 
    'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8',
    'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 
    'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8','coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9',
    'coif10', 'coif11','coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17','sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7',
    'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20']

    db = ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14','db15', 
    'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28',
    'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38']
    bior = ['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8']
    rbio = ['rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8']
    coif = ['coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11','coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17']
    sym = ['sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20']

    for i in wavelet:
        dec = pywt.wavedec(raw,i,level=levels) 
        arr_ratio = []
        arr_Etot = []
        arr_S = []
        for i in dec[1:]:
            ratio, Etot, Stot, E = shannon(i)
            arr_ratio.append(ratio)
            arr_Etot.append(Etot)
            arr_S.append(Stot)
        sum_ratio = np.sum(arr_ratio)
        sum_Etot = np.sum(arr_Etot)
        sum_S = np.sum(arr_S)
        ave_ratio = sum_ratio/levels
        ave_Etot = sum_Etot/levels
        ave_S = sum_S/levels
        arrJ.append(ave_ratio)
        arrJ_Etot.append(ave_Etot)
        arrJ_S.append(ave_S)
        
    #arr_isntNaN = [x for x in arrJ if str(x) != 'nan']
    
    max_arr = [i for i in range(0, len(np.nan_to_num(arrJ))) if arrJ[i] == np.max(np.nan_to_num(arrJ))]
    min_arrEtot = [i for i in range(0, len(np.nan_to_num(arrJ_Etot))) if arrJ_Etot[i] == np.min(np.nan_to_num(arrJ_Etot))]
    Energy_to_Shannon_Entropy = np.nan_to_num(arrJ)
    Energy = np.nan_to_num(arrJ_Etot)
    Shannon_Entropy = np.nan_to_num(arrJ_S)

    result = (f'The first of wavelet:{wavelet[max_arr[0]]}')
    result_min = (f'The min of Etot:{wavelet[min_arrEtot[0]]}')

    value = {'E-to-S':Energy_to_Shannon_Entropy}

    s = pd.DataFrame(value, index=[i for i in wavelet])

    return s, result, result_min, wavelet, Energy_to_Shannon_Entropy, Energy, Shannon_Entropy
#===========================================================================================================================#

def NumRectifier(dt=None, ydata=None, kValue=4, R=1e+10, C=1e-12):
    r"""
    This function developed by Szu-Wei et al. at AcroMass technologies Inc., Hukou, Hsinchu, Taiwan 30352, Republic of China
    https://www.researchgate.net/publication/333417539_Charge-sensing_particle_detector_CSPD_a_sensitivity-enhanced_Faraday_cup
    """
    if dt is None or ydata is None:
        return []

    ratio = 1/kValue

    kernelSize = 5
    kernel = np.zeros(kernelSize)
    halflen, rem = divmod(kernelSize, 2)

    kernel[0] = 1 * C / (12 * dt)
    kernel[1] = -2 * C / (3 * dt)
    kernel[2] = -1 / R
    kernel[3] = 2 * C / (3 * dt)
    kernel[4] = -1 * C / (12 * dt)
    kernel = kernel * ratio

    arr=[]
    I_b = np.convolve(ydata, kernel, 'same')
    I_b[0:halflen] = I_b[halflen + 1]
    I_b[len(I_b) - halflen:len(I_b)] = I_b[len(I_b) - halflen - 1]
    return I_b

def convertQV(rec,time,winwidth):
    r"""
    This function developed by Szu-Wei et al. at AcroMass technologies Inc., Hukou, Hsinchu, Taiwan 30352, Republic of China
    https://www.researchgate.net/publication/333417539_Charge-sensing_particle_detector_CSPD_a_sensitivity-enhanced_Faraday_cup
    """
    rec = NumRectifier(time,rec)
    winsize = np.floor(winwidth/time)
    id = [i - winsize for i in range(int(winsize))]
    kernel = [np.exp((-i**2)/(2*(winsize/6)**2)) for i in id]
    kernel = [i/sum(kernel) for i in kernel]
    #sum = np.sum(kernel)
    rec = np.convolve(rec, kernel, 'same')
    w = int(np.floor(winsize))
    L = len(rec)
    for i in range(w):
        rec[i] = rec[w]
        rec[L-i-1] = rec[L-w-1]
    return rec

#====================================================================================================================================#

def top_hat_baseline(data, w, c, step):
    r"""
    Code created based on Stanford's paper et al., 2016!
    Stanford's paper: DOI 10.1186/s12953-016-0107-8
    data: Input your data;
    w: Estimate width of peak in mass spectrum;
    c: Convergence parameter, if c = 1 is convergence above, elif c = -1 is convergence below;
    step: it is used to create a kernel array of [-w, w+1, step].
    
    """
#Boundaries condition#

    if w%2 != 0:
        raise TypeError("w is a even number!")
    elif w <= 0:
        raise TypeError("w is a positive number and larger zero!")

#Create erosions and dilations values#

    def inf_sup(input, w):
        
        epsilon = []
        omega = []
        B = np.arange(-w,w+1,step)

        for i in range(0,len(input),1):
            erosion = B + i
            for i in range(0,len(erosion)):
                if erosion[i] < 0:
                    erosion[i] = 0
                else:
                    break
            if erosion[0] < 0:
                erosion[0] = 0
            filter_e = input[erosion[0]:erosion[-1]+1]
            epsilon.append(np.min(filter_e))

        for i in range(0,len(epsilon),1):
            dilation = B + i
            for i in range(0,len(dilation)):
                if dilation[i] < 0:
                    dilation[i] = 0
                else:
                    break
            if dilation[0] < 0:
                dilation[0] = 0
            filter_d = epsilon[dilation[0]:dilation[-1]+1]
            omega.append(np.max(filter_d))

        #if input[0] > omega[0]:
            
        omega[0] = epsilon[0] = input[0] # -> Modify data

        return np.asarray(epsilon), np.asarray(omega)

    epsilon, omega = inf_sup(data, w)

#Smoothly baseline#

    def smooth_baseline(epsilon, omega, c):
        
        ave = (omega + epsilon)/2
        
        if c == 1: #converge above
            baseline = (3*omega + ave)/4
        if c == -1: #converge below
            baseline = (3*epsilon + ave)/4
        
        return baseline
    baseline = smooth_baseline(epsilon, omega, c)

#Substract basline#

    substract = data - baseline
    
    return epsilon, omega, baseline, substract