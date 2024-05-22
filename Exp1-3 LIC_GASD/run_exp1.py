
try: from pycamia import info_manager
except ImportError: print("Warning: pycamia not loaded. ")

__info__ = info_manager(
    project = "",
    package = "",
    author = "", 
    create = "",
    fileinfo = "",
    requires = ""
)

import os, sys, re
import batorch as bt
import micomputing as mc
from pycamia import *

def main():
    for exp in """
        #T2_NVI
        #T2_MI
        #T2_NMI
        #T2_NCC
        #T2_NVI_local
        #T2_MI_local
        #T2_NCC_local
        #T2_GASD
        #T2_LIC
        #T1_LIC_sig0
        #T1_LIC_sig0_01
        #T1_LIC_sig0_1
        #T1_LIC_sig1
        #T1_LIC_sig10
        #T1_LIC_noise_sig0
        #T1_LIC_noise_sig0_01
        #T1_LIC_noise_sig0_1
        #T1_LIC_noise_sig1
        #T1_LIC_noise_sig10
        #T1_CTr_GASD_sig0
        #T1_CTr_GASD_sig0_01
        #T1_CTr_GASD_sig0_1
        #T1_CTr_GASD_sig1
        #T1_CTr_GASD_sig10
        T1_CTr_GASD_noise_sig0
        T1_CTr_GASD_noise_sig0_01
        T1_CTr_GASD_noise_sig0_1
        T1_CTr_GASD_noise_sig1
        T1_CTr_GASD_noise_sig10
        #T1_CTr_LIC_sig0
        #T1_CTr_LIC_sig0_01
        #T1_CTr_LIC_sig0_1
        #T1_CTr_LIC_sig1
        #T1_CTr_LIC_sig10
        T1_CTr_LIC_noise_sig0
        T1_CTr_LIC_noise_sig0_01
        T1_CTr_LIC_noise_sig0_1
        T1_CTr_LIC_noise_sig1
        T1_CTr_LIC_noise_sig10
    """.split():
        if exp.startswith('#'): continue
        os.system(f"python3 Exp1_LIC_GASD.py --exp {exp}")

if __name__ == "__main__": main()
