#!/usr/bin/env python
# coding: utf-8

# In[725]:

import sys
import sqlite3
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import sympy as sym
from sympy.utilities.lambdify import lambdify
from sympy.utilities.iterables import flatten
from sympy import sympify
import matplotlib.pyplot as plt
import math
from copy import deepcopy
import random
from functools import cmp_to_key
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as optimize


# ### Input  

# In[726]:


def validCoupling(coup):
    return len(coup) == 5 and coup[:2] == "LM" and coup[2] in ['1','2','3'] and coup[3] in ['1','2','3'] and coup[4] in ['L','R']


# In[727]:

mass = 0
lambdastring = []
lam_vals = []
ignorePairSingle = False
with open('input.txt') as f:
    contents = f.readlines()
    try:
        mass = float(contents[0])
    except:
        print("Mass input is non-numeric. Please check again.")
        sys.exit()
    
    lambdastring = contents[1].strip().split(' ')
    for coup in lambdastring:
        if(validCoupling(coup) == False):
            print("Invalid coupling format.")
            sys.exit()
    if contents[2].strip().lower() == "yes":
        ignorePairSingle = True
    
with open('values.txt') as f:
    contents = f.readlines()    
    for content in contents:
        lam_vals.append(content.strip().split(' '))
    for lam_vals_index in lam_vals:    
        for lam_val in lam_vals_index:
            try:
                temp_var = float(lam_val)
            except:
                print("Coupling value is not numerical")
                sys.exit()

num_lam = len(lambdastring)

# ### Sort the lambdas

# In[728]:


def compare_lambda(item1, item2):
    a1 = list(item1[0])
    a2 = list(item2[0])
    if a1[3] != a2[3]:
        return ord(a1[3]) - ord(a2[3])
    else:
        if a1[4] != a2[4]:
            if a1[4] == 'L':
                return -1
            return 1
        else:
            return ord(a1[2]) - ord(a2[2])
    return 0


# In[729]:

temp_lam_vals = []
original_lambdastring = lambdastring
for lam_val in lam_vals:
    combined_lambda = zip(original_lambdastring, lam_val)
    combined_lambda = sorted(combined_lambda, key=cmp_to_key(compare_lambda))
    combined_lambda = list(zip(*combined_lambda))
    lambdastring = list(combined_lambda[0])
    temp_lam_vals.append(list(combined_lambda[1]))

lam_vals_original = lam_vals
lam_vals = temp_lam_vals
lam = sym.symbols(lambdastring)
chi_sq_limits = [4.00, 6.17, 8.02488, 9.7156, 11.3138, 12.8488]

# connect to the database expressions.db
print("Creating expressions.db if it does not already exist")
sqliteConnection = sqlite3.connect('expressions.db')
cursor = sqliteConnection.cursor()

listOfTables = cursor.execute(
  """SELECT name FROM sqlite_master WHERE type='table'
  AND name='couplings'; """).fetchall()

# check if table exists
if listOfTables == []:
    print("Table created")
    cursor.execute("""CREATE TABLE couplings(MASS FLOAT, IGNORE_S_P BOOLEAN,LAMBDASTRING CHAR, CHI_SQ_EXPRESSION TEXT, CHI_SQ_MIN FLOAT);""")

# check if inputted values are already in the database
queryResult = cursor.execute(""" SELECT * FROM couplings 
    WHERE MASS = ? and  IGNORE_S_P = ? AND LAMBDASTRING = ? """,(mass,ignorePairSingle," ".join(lambdastring))).fetchall()
if queryResult != []:
    print("Getting expression from database")
    chisq_min = queryResult[0][4]
    chisq_symb = sympify(queryResult[0][3])
    numpy_chisq=lambdify(flatten(lam),chisq_symb, modules='numpy')
    for lam_val_original in lam_vals_original:
        tempDict = dict(zip(original_lambdastring, lam_val_original))
        sortedLamVals = [tempDict.get(x) for x in lambdastring]
        print(lam_val_original)
        temp = [float(x) for x in sortedLamVals]
        all_zeroes = True
        for x in temp:
            if x:
                all_zeroes = False
                break
        if all_zeroes:
            print("Yes")
            continue
        chisq_given_vals = numpy_chisq(*flatten(temp))
        if chisq_given_vals - chisq_min <= chi_sq_limits[len(lambdastring)-1]:
            print("Yes")
        else:
            print("No")
    sqliteConnection.commit() # commit changes
    sqliteConnection.close() # close connection 
    sys.exit()


# ## File paths

# In[730]:


cs_sc_path="./data/cross_section/"
df_pair = pd.read_csv(cs_sc_path + "pair.csv")
df_single = pd.read_csv(cs_sc_path + "single.csv")
df_interference = pd.read_csv(cs_sc_path + "interference.csv")
df_tchannel = pd.read_csv(cs_sc_path + "tchannel.csv")
df_pureqcd = pd.read_csv(cs_sc_path + "pureqcd.csv")


# In[731]:


cross_terms_tchannel = "./data/cross_section/tchannel_doublecoupling.csv"
double_coupling_data_tchannel = pd.read_csv(cross_terms_tchannel, header=[0])
ee_t_ct = [double_coupling_data_tchannel[lambdastring[i] + '_' + lambdastring[j]] for i in range(num_lam) for j in range(i+1, num_lam) if lambdastring[i][3] == lambdastring[j][3] and lambdastring[i][3] == '1']
mumu_t_ct = [double_coupling_data_tchannel[lambdastring[i] + '_' + lambdastring[j]] for i in range(num_lam) for j in range(i+1, num_lam) if lambdastring[i][3] == lambdastring[j][3] and lambdastring[i][3] == '2']
tautau_t_ct = [double_coupling_data_tchannel[lambdastring[i] + '_' + lambdastring[j]] for i in range(num_lam) for j in range(i+1, num_lam) if lambdastring[i][3] == lambdastring[j][3] and lambdastring[i][3] == '3']


# In[732]:


data_mass_list = [1000, 1500, 2000, 2500, 3000]

interpolation_type='slinear'
def interpolate_cs_func(df, ls=lambdastring):
    return lambda mass: [interp1d(data_mass_list, df[coupling][:5], kind=interpolation_type)([mass])[0] for coupling in ls]

def interpolate_cs_ct_func(df):
    return lambda mass: [interp1d(data_mass_list, df[ij], kind=interpolation_type)([mass])[0] for ij in range(len(df)) ]

# f = interpolate_cs_t_ct_func(df_t_ct)
# f(1100)


# In[733]:


cs_q = interpolate_cs_func(df_pureqcd)
cs_p = interpolate_cs_func(df_pair)
cs_s = interpolate_cs_func(df_single)
cs_i = interpolate_cs_func(df_interference)
cs_t = interpolate_cs_func(df_tchannel)

cs_l = [cs_q(mass), cs_p(mass), cs_s(mass), cs_i(mass), cs_t(mass)]
ee_cs = []
mumu_cs = []
tautau_cs = []

for process in cs_l:
    ee_temp = []
    mumu_temp = []
    tautau_temp = []
    for lamda,cs in zip(lambdastring,process):
        if lamda[3] == '1':
            ee_temp.append(cs)
        elif lamda[3] == '2':
            mumu_temp.append(cs)
        elif lamda[3] == '3':
            tautau_temp.append(cs)
    ee_cs.append(ee_temp)
    mumu_cs.append(mumu_temp)
    tautau_cs.append(tautau_temp)

cs_ee_t_ct_func = interpolate_cs_ct_func(ee_t_ct)
cs_ee_t_ct_temp = cs_ee_t_ct_func(mass)
cs_mumu_t_ct_func = interpolate_cs_ct_func(mumu_t_ct)
cs_mumu_t_ct_temp = cs_mumu_t_ct_func(mass)
cs_tautau_t_ct_func = interpolate_cs_ct_func(tautau_t_ct)
cs_tautau_t_ct_temp = cs_tautau_t_ct_func(mass)


# In[734]:


ee_cntr = 0
cs_ee_t_ct = cs_ee_t_ct_temp[:]
mumu_cntr = 0
cs_mumu_t_ct = cs_mumu_t_ct_temp[:]
tautau_cntr = 0
cs_tautau_t_ct = cs_tautau_t_ct_temp[:]

for i in range(num_lam):
    for j in range(i+1, num_lam):
        if lambdastring[i][3] == lambdastring[j][3]:
            if lambdastring[i][3] == '1':
                cs_ee_t_ct[ee_cntr] = cs_ee_t_ct_temp[ee_cntr] - cs_l[4][i] - cs_l[4][j]
                ee_cntr += 1
            elif lambdastring[i][3] == '2':
                cs_mumu_t_ct[mumu_cntr] = cs_mumu_t_ct_temp[mumu_cntr] - cs_l[4][i] - cs_l[4][j]
                mumu_cntr += 1
            elif lambdastring[i][3] == '3':
                cs_tautau_t_ct[tautau_cntr] = cs_tautau_t_ct_temp[tautau_cntr] - cs_l[4][i] - cs_l[4][j]
                tautau_cntr += 1

# ## find closest mass for efficiencies

# In[735]:


closest_mass = 0
closest_diff = 10000
for val in data_mass_list:
    if abs(mass-val) < closest_diff:
        closest_diff = abs(mass-val)
        closest_mass = val


# ### Efficiencies file path
# Now that we have the type of coupling and the mass for which we need to calculate the coupling, we can get the efficiency of the closest masspoint for which efficiency has been calculated. (cross-section was interpolated as a pattern was found among them).
# The paths printed below contai 4 file:
# * HHbT.csv
# * HHbV.csv
# * LHbT.csv
# * LHbV.csv

# In[736]:


efficiency_prefix = "./data/efficiency/"
path_interference_ee = [efficiency_prefix + "i/" + str(coupling[2:]) for coupling in lambdastring if coupling[3]=='1']
path_pair_ee = [efficiency_prefix + "p/" + str(coupling[2:]) for coupling in lambdastring if coupling[3]=='1']
path_single_ee = [efficiency_prefix + "s/" + str(coupling[2:]) for coupling in lambdastring if coupling[3]=='1']
path_tchannel_ee = [efficiency_prefix + "t/" + str(coupling[2:]) for coupling in lambdastring if coupling[3]=='1']
path_pureqcd_ee = [efficiency_prefix + "q/" + str(coupling[2:]) for coupling in lambdastring if coupling[3]=='1']

path_interference_mumu = [efficiency_prefix + "i/" + str(coupling[2:]) for coupling in lambdastring if coupling[3]=='2']
path_pair_mumu = [efficiency_prefix + "p/" + str(coupling[2:]) for coupling in lambdastring if coupling[3]=='2']
path_single_mumu = [efficiency_prefix + "s/" + str(coupling[2:]) for coupling in lambdastring if coupling[3]=='2']
path_tchannel_mumu = [efficiency_prefix + "t/" + str(coupling[2:]) for coupling in lambdastring if coupling[3]=='2']
path_pureqcd_mumu = [efficiency_prefix + "q/" + str(coupling[2:]) for coupling in lambdastring if coupling[3]=='2']

path_interference_tautau = [efficiency_prefix + "i/" + str(coupling[2:]) + "/" + str(closest_mass) for coupling in lambdastring if coupling[3]=='3']
path_pair_tautau = [efficiency_prefix + "p/" + str(coupling[2:]) + "/" + str(closest_mass) for coupling in lambdastring if coupling[3]=='3']
path_single_tautau = [efficiency_prefix + "s/" + str(coupling[2:]) + "/" + str(closest_mass) for coupling in lambdastring if coupling[3]=='3']
path_tchannel_tautau = [efficiency_prefix + "t/" + str(coupling[2:]) + "/" + str(closest_mass) for coupling in lambdastring if coupling[3]=='3']
path_pureqcd_tautau = [efficiency_prefix + "q/" + str(coupling[2:]) + "/" + str(closest_mass) for coupling in lambdastring if coupling[3]=='3']


# In[737]:


t_ct_prefix = "./data/efficiency/t/"

ee_path_t_ct = []
mumu_path_t_ct = []
tautau_path_t_ct = []

for i in range(num_lam):
    for j in range(i+1, num_lam):
        if lambdastring[i][3] == lambdastring[j][3]:
            if lambdastring[i][3] == '1':
                ee_path_t_ct.append(t_ct_prefix + str(lambdastring[i][2:]) + "_" + str(lambdastring[j][2:]) )
            elif lambdastring[i][3] == '2':
                mumu_path_t_ct.append(t_ct_prefix + str(lambdastring[i][2:]) + "_" + str(lambdastring[j][2:]) )
            elif lambdastring[i][3] == '3':
                tautau_path_t_ct.append(t_ct_prefix + str(lambdastring[i][2:]) + "_" + str(lambdastring[j][2:]) + "/" + str(closest_mass))

# ## Code:

# In[738]:


tagnames = ["/HHbT.csv", "/HHbV.csv", "/LHbT.csv", "/LHbV.csv"]

ee_eff_l = [[[pd.read_csv(path_pureqcd_ee[i] + "/" + str(int(closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(path_pureqcd_ee))],
         [[pd.read_csv(path_pair_ee[i] + "/" + str(int(closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(path_pureqcd_ee))],
         [[pd.read_csv(path_single_ee[i] + "/" + str(int(closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(path_pureqcd_ee))],
         [[pd.read_csv(path_interference_ee[i] + "/" + str(int(closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(path_pureqcd_ee))],
         [[pd.read_csv(path_tchannel_ee[i] + "/" + str(int(closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(path_pureqcd_ee))]]

mumu_eff_l = [[[pd.read_csv(path_pureqcd_mumu[i] + "/" + str(int(closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(path_pureqcd_mumu))],
         [[pd.read_csv(path_pair_mumu[i] + "/" + str(int(closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(path_pureqcd_mumu))],
         [[pd.read_csv(path_single_mumu[i] + "/" + str(int(closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(path_pureqcd_mumu))],
         [[pd.read_csv(path_interference_mumu[i] + "/" + str(int(closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(path_pureqcd_mumu))],
         [[pd.read_csv(path_tchannel_mumu[i] + "/" + str(int(closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(path_pureqcd_mumu))]]


tautau_eff_l = [[[pd.read_csv(path_pureqcd_tautau[i] + j,header=[0]).to_numpy()[:,2] for j in tagnames] for i in range(len(path_pureqcd_tautau))],
         [[pd.read_csv(path_pair_tautau[i] + j,header=[0]).to_numpy()[:,2] for j in tagnames] for i in range(len(path_pureqcd_tautau))],
         [[pd.read_csv(path_single_tautau[i] + j,header=[0]).to_numpy()[:,2] for j in tagnames] for i in range(len(path_pureqcd_tautau))],
         [[pd.read_csv(path_interference_tautau[i] + j,header=[0]).to_numpy()[:,2] for j in tagnames] for i in range(len(path_pureqcd_tautau))],
         [[pd.read_csv(path_tchannel_tautau[i] + j,header=[0]).to_numpy()[:,2] for j in tagnames] for i in range(len(path_pureqcd_tautau))]]

ee_eff_t_ct_temp = [[pd.read_csv(ee_path_t_ct[i] + "/" + str(int(closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(ee_path_t_ct))]
mumu_eff_t_ct_temp = [[pd.read_csv(mumu_path_t_ct[i] + "/" + str(int(closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(mumu_path_t_ct))]
tautau_eff_t_ct_temp = [[pd.read_csv(tautau_path_t_ct[j] + i,header=[0]).to_numpy()[:,2] for i in tagnames]  for j in range(len(tautau_path_t_ct))]


# In[739]:


ee_eff_t_ct = deepcopy(ee_eff_t_ct_temp)
mumu_eff_t_ct = deepcopy(mumu_eff_t_ct_temp)
tautau_eff_t_ct = deepcopy(tautau_eff_t_ct_temp)

ee_cntr = 0
mumu_cntr = 0
tautau_cntr = 0
for i in range(len(path_interference_ee)):
    for j in range(i+1, len(path_interference_ee)):
        ee_eff_t_ct[ee_cntr][0] = (ee_eff_t_ct_temp[ee_cntr][0]*cs_ee_t_ct_temp[ee_cntr] - ee_eff_l[4][i][0]*ee_cs[4][i] - ee_eff_l[4][j][0]*ee_cs[4][j])/cs_ee_t_ct[ee_cntr]
        ee_cntr += 1
for i in range(len(path_interference_mumu)):
    for j in range(i+1, len(path_interference_mumu)):
        mumu_eff_t_ct[mumu_cntr][0] = (mumu_eff_t_ct_temp[mumu_cntr][0]*cs_mumu_t_ct_temp[mumu_cntr] - mumu_eff_l[4][i][0]*mumu_cs[4][i] - mumu_eff_l[4][j][0]*mumu_cs[4][j])/cs_mumu_t_ct[mumu_cntr]
        mumu_cntr += 1
for i in range(len(path_interference_tautau)):
    for j in range(i+1, len(path_interference_tautau)):
        for tag_num in range(4):
            tautau_eff_t_ct[tautau_cntr][tag_num] = (tautau_eff_t_ct_temp[tautau_cntr][tag_num]*cs_tautau_t_ct_temp[tautau_cntr] - tautau_eff_l[4][i][tag_num]*tautau_cs[4][i] - tautau_eff_l[4][j][tag_num]*tautau_cs[4][j])/cs_tautau_t_ct[tautau_cntr]
        tautau_cntr += 1


# In[740]:


standard_HHbT = pd.read_csv('./data/HEPdata/HHbT.csv',header=[0])
standard_HHbV = pd.read_csv('./data/HEPdata/HHbV.csv',header=[0])
standard_LHbT = pd.read_csv('./data/HEPdata/LHbT.csv',header=[0])
standard_LHbV = pd.read_csv('./data/HEPdata/LHbV.csv',header=[0])
standard_ee = pd.read_csv('./data/HEPdata/dielectron.csv',header=[0])
standard_mumu = pd.read_csv('./data/HEPdata/dimuon.csv',header=[0])
ND = [standard_HHbT['ND'].to_numpy(), standard_HHbV['ND'].to_numpy(),
      standard_LHbT['ND'].to_numpy(), standard_LHbV['ND'].to_numpy(),
      standard_ee['ND'].to_numpy(),standard_mumu['ND'].to_numpy()]
NSM = [standard_HHbT['Standard Model'].to_numpy(), standard_HHbV['Standard Model'].to_numpy(),
       standard_LHbT['Standard Model'].to_numpy(), standard_LHbV['Standard Model'].to_numpy(),
      standard_ee['Standard Model'].to_numpy(),standard_mumu['Standard Model'].to_numpy()]


# ### Momentum
# Momentum($M_{leptoquark}, m_{quark}, m_{lepton}$) = $\frac{\sqrt{A*B}}{(2*M_{leptoquark})}$ \
# A = $(M_{leptoquark} + m_{lepton})^2 - m_{quark}^2$ \
# B = $(M_{leptoquark} - m_{lepton})^2 - m_{quark}^2$
# 
# ### Abseffcoupl
# Abseffcoupl($\lambda,M_{lq}, m_q, m_l$) = $\lambda^2 *$ abseffcoupl_massfactor \
# abseffcoupl_massfactor = $[M_{lq}^2 - (m_l^2 + mq^2) - \frac{(ml^2-mq^2)^2}{M_{lq}^2} - (6 * m_l * m_q)]$

# In[741]:


def momentum(Mlq, mq, ml):
    a = (Mlq + ml)**2 - mq**2
    b = (Mlq - ml)**2 - mq**2
    return math.sqrt(a*b)/(2*Mlq)

def abseffcoupl_massfactor(Mlq, mq, ml):
    return Mlq**2 - (ml**2 + mq**2) - (ml**2 - mq**2)**2/Mlq**2 - (6*ml*mq)

def decay_width_massfactor(Mlq, M):
    #M is a list with [Mlq, mq, ml]
    return momentum(Mlq,M[0], M[1])*abseffcoupl_massfactor(Mlq,M[0],M[1])/(8 * math.pi**2 * Mlq**2)


# In[742]:


mev2gev = 0.001

mass_quarks = {'1': [2.3 , 4.8], '2': [1275, 95], '3': [173070, 4180]}
mass_leptons= {'1': [0.511, 0.0022], '2': [105.7, 0.17], '3': [1777, 15.5]}

for gen in mass_quarks:
    mass_quarks[gen] = [x*mev2gev for x in mass_quarks[gen]]

for gen in mass_leptons:
    mass_leptons[gen] = [x*mev2gev for x in mass_leptons[gen]]

def make_mass_dict(ls = lambdastring):
    md = {}
    for i in range(num_lam):
        if ls[i][4] == 'L':
            md[lambdastring[i]] = [[mass_quarks[ls[i][2]][1], mass_leptons[ls[i][3]][0]], [mass_quarks[ls[i][2]][0], mass_leptons[ls[i][3]][1]]]
        elif ls[i][4] == 'R':
            md[lambdastring[i]] = [[mass_quarks[ls[i][2]][1], mass_leptons[ls[i][3]][0]]]
    return md

mass_dict = make_mass_dict()

def branching_fraction(ls = lambdastring, lm = lam, md = mass_dict, Mlq = mass, width_const = 0):
    denom = 0
    numer = 0
    for i in range(len(ls)):
        denom += lm[i]**2 * decay_width_massfactor(Mlq, mass_dict[ls[i]][0])
        numer += lm[i]**2 * decay_width_massfactor(Mlq, mass_dict[ls[i]][0])
        if ls[i][4] == 'L':
            denom += lm[i]**2 * decay_width_massfactor(Mlq, md[ls[i]][1])
    return numer/(denom + width_const)


# In[743]:


luminosity = 139 * 1000

b_frac = branching_fraction()

ee_lambdas_len = len(path_pureqcd_ee)
mumu_lambdas_len = len(path_pureqcd_mumu)
tautau_lambdas_len = len(path_pureqcd_tautau)

ee_lam = []
mumu_lam = []
tautau_lam = []

for lamda in lam:
    temp_str_sym = str(lamda)
    if temp_str_sym[3] == '1':
        ee_lam.append(lamda)
    elif temp_str_sym[3] == '2':
        mumu_lam.append(lamda)
    elif temp_str_sym[3] == '3':
        tautau_lam.append(lamda)
        
print("Computing chi-square expression ... ")

def get_chisq_ind(tag):
    if (tag<4 and len(tautau_eff_l[0])==0) or (tag==4 and len(ee_eff_l[0])==0) or (tag==5 and len(mumu_eff_l[0])==0):
        return 0
    if tag<4:
        num_bin = len(tautau_eff_l[0][0][tag])
    elif tag == 4:
        num_bin = len(ee_eff_l[0][0][0])
    elif tag == 5:
        num_bin = len(mumu_eff_l[0][0][0])
    nq = [0.0]*num_bin
    np = [0.0]*num_bin
    ns = [0.0]*num_bin
    ni = [0.0]*num_bin
    nt = [0.0]*num_bin
    ntc= [0.0]*num_bin
    nsm= NSM[tag]
    nd = ND[tag]
    denominator = [nd[bin_no] + 0.01*nd[bin_no]**2 for bin_no in range(num_bin)]
    if tag<4:
        nq = [nq[bin_no] + tautau_cs[0][0]*tautau_eff_l[0][0][tag][bin_no] * b_frac**2 *luminosity for bin_no in range(num_bin)]
        for i in range(tautau_lambdas_len):
            np = [np[bin_no] + tautau_cs[1][i]*tautau_eff_l[1][i][tag][bin_no]*tautau_lam[i]**4 *b_frac**2 *luminosity for bin_no in range(num_bin)]
            ns = [ns[bin_no] + tautau_cs[2][i]*tautau_eff_l[2][i][tag][bin_no]*tautau_lam[i]**2 *b_frac    *luminosity for bin_no in range(num_bin)]
            ni = [ni[bin_no] + tautau_cs[3][i]*tautau_eff_l[3][i][tag][bin_no]*tautau_lam[i]**2               *luminosity for bin_no in range(num_bin)]
            nt = [nt[bin_no] + tautau_cs[4][i]*tautau_eff_l[4][i][tag][bin_no]*tautau_lam[i]**4               *luminosity for bin_no in range(num_bin)]
        ntc_cntr = 0
        for i in range(tautau_lambdas_len):
            for j in range(i+1, tautau_lambdas_len):
                "use cross-terms"
                ntc = [ntc[bin_no] + cs_tautau_t_ct[ntc_cntr]*tautau_eff_t_ct[ntc_cntr][tag][bin_no]* tautau_lam[i]**2 * tautau_lam[j]**2            *luminosity for bin_no in range(num_bin)]
                ntc_cntr += 1
    elif tag==4:
        nq = [nq[bin_no] + ee_cs[0][0]*ee_eff_l[0][0][0][bin_no] * b_frac**2 *luminosity for bin_no in range(num_bin)]
        for i in range(ee_lambdas_len):
            np = [np[bin_no] + ee_cs[1][i]*ee_eff_l[1][i][0][bin_no]*ee_lam[i]**4 *b_frac**2 *luminosity for bin_no in range(num_bin)]
            ns = [ns[bin_no] + ee_cs[2][i]*ee_eff_l[2][i][0][bin_no]*ee_lam[i]**2 *b_frac    *luminosity for bin_no in range(num_bin)]
            ni = [ni[bin_no] + ee_cs[3][i]*ee_eff_l[3][i][0][bin_no]*ee_lam[i]**2               *luminosity for bin_no in range(num_bin)]
            nt = [nt[bin_no] + ee_cs[4][i]*ee_eff_l[4][i][0][bin_no]*ee_lam[i]**4               *luminosity for bin_no in range(num_bin)]
        ntc_cntr = 0
        for i in range(ee_lambdas_len):
            for j in range(i+1, ee_lambdas_len):
                "use cross-terms"
                ntc = [ntc[bin_no] + cs_ee_t_ct[ntc_cntr]*ee_eff_t_ct[ntc_cntr][0][bin_no]* ee_lam[i]**2 * ee_lam[j]**2            *luminosity for bin_no in range(num_bin)]
                ntc_cntr += 1
    elif tag==5:
        nq = [nq[bin_no] + mumu_cs[0][0]*mumu_eff_l[0][0][0][bin_no] * b_frac**2 *luminosity for bin_no in range(num_bin)]
        for i in range(mumu_lambdas_len):
            np = [np[bin_no] + mumu_cs[1][i]*mumu_eff_l[1][i][0][bin_no]*mumu_lam[i]**4 *b_frac**2 *luminosity for bin_no in range(num_bin)]
            ns = [ns[bin_no] + mumu_cs[2][i]*mumu_eff_l[2][i][0][bin_no]*mumu_lam[i]**2 *b_frac    *luminosity for bin_no in range(num_bin)]
            ni = [ni[bin_no] + mumu_cs[3][i]*mumu_eff_l[3][i][0][bin_no]*mumu_lam[i]**2            *luminosity for bin_no in range(num_bin)]
            nt = [nt[bin_no] + mumu_cs[4][i]*mumu_eff_l[4][i][0][bin_no]*mumu_lam[i]**4            *luminosity for bin_no in range(num_bin)]
        ntc_cntr = 0
        for i in range(mumu_lambdas_len):
            for j in range(i+1, mumu_lambdas_len):
                "use cross-terms"
                ntc = [ntc[bin_no] + cs_mumu_t_ct[ntc_cntr]*mumu_eff_t_ct[ntc_cntr][0][bin_no]* mumu_lam[i]**2 * mumu_lam[j]**2            *luminosity for bin_no in range(num_bin)]
                ntc_cntr += 1
    chi_ind = 0.0
    for bin_no in range(num_bin):
        if ignorePairSingle:
            chi_ind += ((nq[bin_no]+ni[bin_no]+nt[bin_no]+ntc[bin_no]+nsm[bin_no]-nd[bin_no])**2)/denominator[bin_no] 
        else:
            chi_ind += ((nq[bin_no]+np[bin_no]+ns[bin_no]+ni[bin_no]+nt[bin_no]+ntc[bin_no]+nsm[bin_no]-nd[bin_no])**2)/denominator[bin_no] 
    chi_sq_tag = sym.Add(chi_ind)
    return chi_sq_tag
    
# c = get_chisq_ind(3)
def get_chi_square_symb():    
    ee_chi = 0
    mumu_chi = 0
    hhbt_chi = 0
    hhbv_chi = 0
    lhbt_chi = 0
    lhbv_chi = 0
    ee_chi = sym.simplify(get_chisq_ind(4))
    print("ee done.")
    mumu_chi = sym.simplify(get_chisq_ind(5))
    print("mumu done.")
    hhbt_chi = sym.simplify(get_chisq_ind(0))
    print("HHbT done.")
    hhbv_chi = sym.simplify(get_chisq_ind(1))
    print("HHbV done.")
    lhbt_chi = sym.simplify(get_chisq_ind(2))    
    print("LHbT done.")
    lhbv_chi = sym.simplify(get_chisq_ind(3))
    print("LHbV done.")
    return sym.Add(ee_chi,mumu_chi,hhbt_chi, hhbv_chi, lhbt_chi, lhbv_chi)

chisq_symb = sym.simplify(get_chi_square_symb())
# temp_str = str(chisq_symb)
# chisq_symb = sympify(temp_str)
numpy_chisq=lambdify(flatten(lam),chisq_symb, modules='numpy')


# In[744]:

print("Computing chi-square minima ... ")
def f(x):
    z = numpy_chisq(*flatten(x))
    return z

startLambda = 0.5
startLambdas = np.array([startLambda for x in range(num_lam)])
minima = optimize.minimize(f,startLambdas, method='Nelder-Mead',options={'fatol':0.01})


# In[745]:


chisq_min = minima.fun

# update database
cursor.execute("""INSERT INTO couplings VALUES (?,?,?,?,?)""", (mass,ignorePairSingle," ".join(lambdastring),str(chisq_symb),chisq_min) )
sqliteConnection.commit() # commit changes
sqliteConnection.close() # close connection

for lam_val,lam_val_copy in zip(lam_vals,lam_vals_original):
    temp = [float(x) for x in lam_val]
    print(lam_val_copy)
    all_zeroes = True
    for x in temp:
        if x:
            all_zeroes = False
            break
    if all_zeroes:
        print("Yes")
        continue
    chisq_given_vals = numpy_chisq(*flatten(temp))
    if chisq_given_vals - chisq_min <= chi_sq_limits[len(lambdastring)-1]:
        print("Yes")
    else:
        print("No")

    