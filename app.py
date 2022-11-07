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


class Calculator:
    def __init__(self):
        # constants
        self.chi_sq_limits = [4.00, 6.17, 8.02488, 9.7156, 11.3138, 12.8488]
        self.data_mass_list = [1000, 1500, 2000, 2500, 3000]
        self.tagnames = ["/HHbT.csv", "/HHbV.csv", "/LHbT.csv", "/LHbV.csv"]
        self.luminosity = 139 * 1000
        self.interpolation_type='slinear'
        # variables
        self.mass = 0
        self.lambdastring = []
        self.original_lambdastring = []
        self.lam_vals = []
        self.lam_vals_original = []
        self.ignorePairSingle = False
        self.num_lam = 0
        self.lam = None
        self.chisq_min = 1000007
        self.chisq_symb = None
        self.numpy_chisq = None
        self.chisq_given_vals = None
        self.sqliteConnection = None
        self.cursor = None
        self.ee_cs = []
        self.mumu_cs = []
        self.tautau_cs = []
        self.cs_ee_t_ct = []
        self.cs_mumu_t_ct = []
        self.cs_tautau_t_ct = []
        self.closest_mass = 0
        self.ee_eff_l = []
        self.mumu_eff_l = []
        self.tautau_eff_l = []
        self.ee_lambdas_len = 0
        self.mumu_lambdas_len = 0
        self.tautau_lambdas_len = 0
        self.ee_eff_t_ct = []
        self.mumu_eff_t_ct = []
        self.tautau_eff_t_ct = []
        self.ND = []
        self.NSM = []
        self.b_frac = 0

    # to check if the coupling input is valid
    def validCoupling(self, coup):
        return len(coup) == 5 and coup[:2] == "LM" and coup[2] in ['1','2','3'] and coup[3] in ['1','2','3'] and coup[4] in ['L','R']

    # take values as input
    def inputValues(self):
        with open('input.txt') as f:
            contents = f.readlines()
            # assign mass
            try:
                self.mass = float(contents[0])
            except:
                print("Mass input is non-numeric. Please check again.")
                sys.exit()

            # assign couplings
            self.lambdastring = contents[1].strip().split(' ')
            for coup in self.lambdastring:
                if(self.validCoupling(coup) == False):
                    print("Invalid coupling format.")
                    sys.exit()
            if contents[2].strip().lower() == "yes":
                self.ignorePairSingle = True

        # assign coupling values
        with open('values.txt') as f:
            contents = f.readlines()    
            for content in contents:
                self.lam_vals.append(content.strip().split(' '))
            for lam_vals_index in self.lam_vals:    
                for lam_val in lam_vals_index:
                    try:
                        temp_var = float(lam_val)
                    except:
                        print("Coupling value is not numerical")
                        sys.exit()
        
        # assign number of lambdas
        self.num_lam = len(self.lambdastring)


    # Sort the lambdas; not called from main
    def compare_lambda(self, item1, item2):
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

    # copy input to variables that are used within the code
    def convertLambda(self):
        temp_lam_vals = []
        self.original_lambdastring = self.lambdastring
        for lam_val in self.lam_vals:
            combined_lambda = zip(self.original_lambdastring, lam_val)
            combined_lambda = sorted(combined_lambda, key=cmp_to_key(self.compare_lambda))
            combined_lambda = list(zip(*combined_lambda))
            self.lambdastring = list(combined_lambda[0])
            temp_lam_vals.append(list(combined_lambda[1]))

        self.lam_vals_original = self.lam_vals
        self.lam_vals = temp_lam_vals
        self.lam = sym.symbols(self.lambdastring)

    # connect to the database expressions.db
    def connectDatabase(self):
        print("Creating expressions.db if it does not already exist")
        self.sqliteConnection = sqlite3.connect('expressions.db')
        self.cursor = self.sqliteConnection.cursor()

        listOfTables = self.cursor.execute(
          """SELECT name FROM sqlite_master WHERE type='table'
          AND name='couplings'; """).fetchall()

        # check if table exists, & create if it doesn't
        if listOfTables == []:
            print("Table created")
            self.cursor.execute("""CREATE TABLE couplings(MASS FLOAT, IGNORE_S_P BOOLEAN,LAMBDASTRING CHAR, CHI_SQ_EXPRESSION TEXT, CHI_SQ_MIN FLOAT);""")

    # check if inputted values are already in the database
    def checkDatabase(self):
        queryResult = self.cursor.execute(""" SELECT * FROM couplings 
            WHERE MASS = ? and  IGNORE_S_P = ? AND LAMBDASTRING = ? """,(self.mass,self.ignorePairSingle," ".join(self.lambdastring))).fetchall()
        if queryResult != []:
            print("Getting expression from database")
            self.chisq_min = queryResult[0][4]
            self.chisq_symb = sympify(queryResult[0][3])
            self.numpy_chisq=lambdify(flatten(self.lam),self.chisq_symb, modules='numpy')
            for lam_val_original in self.lam_vals_original:
                tempDict = dict(zip(self.original_lambdastring, lam_val_original))
                sortedLamVals = [tempDict.get(x) for x in self.lambdastring]
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
                self.chisq_given_vals = self.numpy_chisq(*flatten(temp))
                if self.chisq_given_vals - self.chisq_min <= self.chi_sq_limits[len(self.lambdastring)-1]:
                    print("Yes")
                else:
                    print("No")
            self.sqliteConnection.commit() # commit changes
            self.sqliteConnection.close() # close connection 
            sys.exit()


    # interpolation function for single coupling; not called from main
    def interpolate_cs_func(self, df):
        return lambda mass: [interp1d(self.data_mass_list, df[coupling][:5], kind=self.interpolation_type)([mass])[0] for coupling in self.lambdastring]

    # interpolation function for cross-terms; not called from main
    def interpolate_cs_ct_func(self, df):
        return lambda mass: [interp1d(self.data_mass_list, df[ij], kind=self.interpolation_type)([mass])[0] for ij in range(len(df)) ]

    #// store cross-section file paths
    def setInterpolatedCrossSection(self):
        # read cross-section .csv files
        cs_sc_path="./data/cross_section/"
        df_pair = pd.read_csv(cs_sc_path + "pair.csv")
        df_single = pd.read_csv(cs_sc_path + "single.csv")
        df_interference = pd.read_csv(cs_sc_path + "interference.csv")
        df_tchannel = pd.read_csv(cs_sc_path + "tchannel.csv")
        df_pureqcd = pd.read_csv(cs_sc_path + "pureqcd.csv")

        # read cross-section of cross-terms
        cross_terms_tchannel = "./data/cross_section/tchannel_doublecoupling.csv"
        double_coupling_data_tchannel = pd.read_csv(cross_terms_tchannel, header=[0])
        ee_t_ct = [double_coupling_data_tchannel[self.lambdastring[i] + '_' + self.lambdastring[j]] for i in range(self.num_lam) for j in range(i+1, self.num_lam) if self.lambdastring[i][3] == self.lambdastring[j][3] and self.lambdastring[i][3] == '1']
        mumu_t_ct = [double_coupling_data_tchannel[self.lambdastring[i] + '_' + self.lambdastring[j]] for i in range(self.num_lam) for j in range(i+1, self.num_lam) if self.lambdastring[i][3] == self.lambdastring[j][3] and self.lambdastring[i][3] == '2']
        tautau_t_ct = [double_coupling_data_tchannel[self.lambdastring[i] + '_' + self.lambdastring[j]] for i in range(self.num_lam) for j in range(i+1, self.num_lam) if self.lambdastring[i][3] == self.lambdastring[j][3] and self.lambdastring[i][3] == '3']


        # interpolate cross-section for single coupling
        cs_q = self.interpolate_cs_func(df_pureqcd)
        cs_p = self.interpolate_cs_func(df_pair)
        cs_s = self.interpolate_cs_func(df_single)
        cs_i = self.interpolate_cs_func(df_interference)
        cs_t = self.interpolate_cs_func(df_tchannel)
        cs_l = [cs_q(self.mass), cs_p(self.mass), cs_s(self.mass), cs_i(self.mass), cs_t(self.mass)]

        # classify single coupling cross-terms according to particle
        for process in cs_l:
            ee_temp = []
            mumu_temp = []
            tautau_temp = []
            for lamda,cs in zip(self.lambdastring,process):
                if lamda[3] == '1':
                    ee_temp.append(cs)
                elif lamda[3] == '2':
                    mumu_temp.append(cs)
                elif lamda[3] == '3':
                    tautau_temp.append(cs)
            self.ee_cs.append(ee_temp)
            self.mumu_cs.append(mumu_temp)
            self.tautau_cs.append(tautau_temp)

        # interpolate cross-section for cross-terms
        cs_ee_t_ct_func = self.interpolate_cs_ct_func(ee_t_ct)
        cs_ee_t_ct_temp = cs_ee_t_ct_func(self.mass)
        cs_mumu_t_ct_func = self.interpolate_cs_ct_func(mumu_t_ct)
        cs_mumu_t_ct_temp = cs_mumu_t_ct_func(self.mass)
        cs_tautau_t_ct_func = self.interpolate_cs_ct_func(tautau_t_ct)
        cs_tautau_t_ct_temp = cs_tautau_t_ct_func(self.mass)

        # remove cross-section of single-coupling processes from cross-terms cross-section
        ee_cntr = 0
        self.cs_ee_t_ct = cs_ee_t_ct_temp[:]
        mumu_cntr = 0
        self.cs_mumu_t_ct = cs_mumu_t_ct_temp[:]
        tautau_cntr = 0
        self.cs_tautau_t_ct = cs_tautau_t_ct_temp[:]

        for i in range(self.num_lam):
            for j in range(i+1, self.num_lam):
                if self.lambdastring[i][3] == self.lambdastring[j][3]:
                    if self.lambdastring[i][3] == '1':
                        self.cs_ee_t_ct[ee_cntr] = cs_ee_t_ct_temp[ee_cntr] - cs_l[4][i] - cs_l[4][j]
                        ee_cntr += 1
                    elif self.lambdastring[i][3] == '2':
                        self.cs_mumu_t_ct[mumu_cntr] = cs_mumu_t_ct_temp[mumu_cntr] - cs_l[4][i] - cs_l[4][j]
                        mumu_cntr += 1
                    elif self.lambdastring[i][3] == '3':
                        self.cs_tautau_t_ct[tautau_cntr] = cs_tautau_t_ct_temp[tautau_cntr] - cs_l[4][i] - cs_l[4][j]
                        tautau_cntr += 1
        
        return cs_ee_t_ct_temp, cs_mumu_t_ct_temp, cs_tautau_t_ct_temp

    # find closest mass for efficiencies
    def setClosestMass(self):
        closest_diff = 10000
        for val in self.data_mass_list:
            if abs(self.mass-val) < closest_diff:
                closest_diff = abs(self.mass-val)
                self.closest_mass = val


    # getting file paths for efficiency values
    def setEfficiency(self, cs_ee_t_ct_temp, cs_mumu_t_ct_temp, cs_tautau_t_ct_temp):
        efficiency_prefix = "./data/efficiency/"
        path_interference_ee = [efficiency_prefix + "i/" + str(coupling[2:]) for coupling in self.lambdastring if coupling[3]=='1']
        path_pair_ee = [efficiency_prefix + "p/" + str(coupling[2:]) for coupling in self.lambdastring if coupling[3]=='1']
        path_single_ee = [efficiency_prefix + "s/" + str(coupling[2:]) for coupling in self.lambdastring if coupling[3]=='1']
        path_tchannel_ee = [efficiency_prefix + "t/" + str(coupling[2:]) for coupling in self.lambdastring if coupling[3]=='1']
        path_pureqcd_ee = [efficiency_prefix + "q/" + str(coupling[2:]) for coupling in self.lambdastring if coupling[3]=='1']

        path_interference_mumu = [efficiency_prefix + "i/" + str(coupling[2:]) for coupling in self.lambdastring if coupling[3]=='2']
        path_pair_mumu = [efficiency_prefix + "p/" + str(coupling[2:]) for coupling in self.lambdastring if coupling[3]=='2']
        path_single_mumu = [efficiency_prefix + "s/" + str(coupling[2:]) for coupling in self.lambdastring if coupling[3]=='2']
        path_tchannel_mumu = [efficiency_prefix + "t/" + str(coupling[2:]) for coupling in self.lambdastring if coupling[3]=='2']
        path_pureqcd_mumu = [efficiency_prefix + "q/" + str(coupling[2:]) for coupling in self.lambdastring if coupling[3]=='2']

        path_interference_tautau = [efficiency_prefix + "i/" + str(coupling[2:]) + "/" + str(self.closest_mass) for coupling in self.lambdastring if coupling[3]=='3']
        path_pair_tautau = [efficiency_prefix + "p/" + str(coupling[2:]) + "/" + str(self.closest_mass) for coupling in self.lambdastring if coupling[3]=='3']
        path_single_tautau = [efficiency_prefix + "s/" + str(coupling[2:]) + "/" + str(self.closest_mass) for coupling in self.lambdastring if coupling[3]=='3']
        path_tchannel_tautau = [efficiency_prefix + "t/" + str(coupling[2:]) + "/" + str(self.closest_mass) for coupling in self.lambdastring if coupling[3]=='3']
        path_pureqcd_tautau = [efficiency_prefix + "q/" + str(coupling[2:]) + "/" + str(self.closest_mass) for coupling in self.lambdastring if coupling[3]=='3']

        #// store lambda values length of each particle
        self.ee_lambdas_len = len(path_pureqcd_ee)
        self.mumu_lambdas_len = len(path_pureqcd_mumu)
        self.tautau_lambdas_len = len(path_pureqcd_tautau)


        # store efficiency in variables
        t_ct_prefix = "./data/efficiency/t/"
        ee_path_t_ct = []
        mumu_path_t_ct = []
        tautau_path_t_ct = []
        for i in range(self.num_lam):
            for j in range(i+1, self.num_lam):
                if self.lambdastring[i][3] == self.lambdastring[j][3]:
                    if self.lambdastring[i][3] == '1':
                        ee_path_t_ct.append(t_ct_prefix + str(self.lambdastring[i][2:]) + "_" + str(self.lambdastring[j][2:]) )
                    elif self.lambdastring[i][3] == '2':
                        mumu_path_t_ct.append(t_ct_prefix + str(self.lambdastring[i][2:]) + "_" + str(self.lambdastring[j][2:]) )
                    elif self.lambdastring[i][3] == '3':
                        tautau_path_t_ct.append(t_ct_prefix + str(self.lambdastring[i][2:]) + "_" + str(self.lambdastring[j][2:]) + "/" + str(self.closest_mass))


        # read efficiency files & store them in variables for single couplings
        self.ee_eff_l = [[[pd.read_csv(path_pureqcd_ee[i] + "/" + str(int(self.closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(path_pureqcd_ee))],
                 [[pd.read_csv(path_pair_ee[i] + "/" + str(int(self.closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(path_pureqcd_ee))],
                 [[pd.read_csv(path_single_ee[i] + "/" + str(int(self.closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(path_pureqcd_ee))],
                 [[pd.read_csv(path_interference_ee[i] + "/" + str(int(self.closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(path_pureqcd_ee))],
                 [[pd.read_csv(path_tchannel_ee[i] + "/" + str(int(self.closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(path_pureqcd_ee))]]

        self.mumu_eff_l = [[[pd.read_csv(path_pureqcd_mumu[i] + "/" + str(int(self.closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(path_pureqcd_mumu))],
                 [[pd.read_csv(path_pair_mumu[i] + "/" + str(int(self.closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(path_pureqcd_mumu))],
                 [[pd.read_csv(path_single_mumu[i] + "/" + str(int(self.closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(path_pureqcd_mumu))],
                 [[pd.read_csv(path_interference_mumu[i] + "/" + str(int(self.closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(path_pureqcd_mumu))],
                 [[pd.read_csv(path_tchannel_mumu[i] + "/" + str(int(self.closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(path_pureqcd_mumu))]]


        self.tautau_eff_l = [[[pd.read_csv(path_pureqcd_tautau[i] + j,header=[0]).to_numpy()[:,2] for j in self.tagnames] for i in range(len(path_pureqcd_tautau))],
                 [[pd.read_csv(path_pair_tautau[i] + j,header=[0]).to_numpy()[:,2] for j in self.tagnames] for i in range(len(path_pureqcd_tautau))],
                 [[pd.read_csv(path_single_tautau[i] + j,header=[0]).to_numpy()[:,2] for j in self.tagnames] for i in range(len(path_pureqcd_tautau))],
                 [[pd.read_csv(path_interference_tautau[i] + j,header=[0]).to_numpy()[:,2] for j in self.tagnames] for i in range(len(path_pureqcd_tautau))],
                 [[pd.read_csv(path_tchannel_tautau[i] + j,header=[0]).to_numpy()[:,2] for j in self.tagnames] for i in range(len(path_pureqcd_tautau))]]

        # read efficiency files & store them in variables for cross-terms
        ee_eff_t_ct_temp = [[pd.read_csv(ee_path_t_ct[i] + "/" + str(int(self.closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(ee_path_t_ct))]
        mumu_eff_t_ct_temp = [[pd.read_csv(mumu_path_t_ct[i] + "/" + str(int(self.closest_mass)) + ".csv",header=[0]).to_numpy()[:,2]] for i in range(len(mumu_path_t_ct))]
        tautau_eff_t_ct_temp = [[pd.read_csv(tautau_path_t_ct[j] + i,header=[0]).to_numpy()[:,2] for i in self.tagnames]  for j in range(len(tautau_path_t_ct))]



        self.ee_eff_t_ct = deepcopy(ee_eff_t_ct_temp)
        self.mumu_eff_t_ct = deepcopy(mumu_eff_t_ct_temp)
        self.tautau_eff_t_ct = deepcopy(tautau_eff_t_ct_temp)

        ee_cntr = 0
        mumu_cntr = 0
        tautau_cntr = 0
        for i in range(len(path_interference_ee)):
            for j in range(i+1, len(path_interference_ee)):
                self.ee_eff_t_ct[ee_cntr][0] = (ee_eff_t_ct_temp[ee_cntr][0]*cs_ee_t_ct_temp[ee_cntr] - self.ee_eff_l[4][i][0]*self.ee_cs[4][i] - self.ee_eff_l[4][j][0]*self.ee_cs[4][j])/self.cs_ee_t_ct[ee_cntr]
                ee_cntr += 1
        for i in range(len(path_interference_mumu)):
            for j in range(i+1, len(path_interference_mumu)):
                self.mumu_eff_t_ct[mumu_cntr][0] = (mumu_eff_t_ct_temp[mumu_cntr][0]*cs_mumu_t_ct_temp[mumu_cntr] - self.mumu_eff_l[4][i][0]*self.mumu_cs[4][i] - self.mumu_eff_l[4][j][0]*self.mumu_cs[4][j])/self.cs_mumu_t_ct[mumu_cntr]
                mumu_cntr += 1
        for i in range(len(path_interference_tautau)):
            for j in range(i+1, len(path_interference_tautau)):
                for tag_num in range(4):
                    self.tautau_eff_t_ct[tautau_cntr][tag_num] = (tautau_eff_t_ct_temp[tautau_cntr][tag_num]*cs_tautau_t_ct_temp[tautau_cntr] - self.tautau_eff_l[4][i][tag_num]*self.tautau_cs[4][i] - self.tautau_eff_l[4][j][tag_num]*self.tautau_cs[4][j])/self.cs_tautau_t_ct[tautau_cntr]
                tautau_cntr += 1


    # read HEP data
    def readHEPData(self):
        standard_HHbT = pd.read_csv('./data/HEPdata/HHbT.csv',header=[0])
        standard_HHbV = pd.read_csv('./data/HEPdata/HHbV.csv',header=[0])
        standard_LHbT = pd.read_csv('./data/HEPdata/LHbT.csv',header=[0])
        standard_LHbV = pd.read_csv('./data/HEPdata/LHbV.csv',header=[0])
        standard_ee = pd.read_csv('./data/HEPdata/dielectron.csv',header=[0])
        standard_mumu = pd.read_csv('./data/HEPdata/dimuon.csv',header=[0])
        self.ND = [standard_HHbT['ND'].to_numpy(), standard_HHbV['ND'].to_numpy(),
              standard_LHbT['ND'].to_numpy(), standard_LHbV['ND'].to_numpy(),
              standard_ee['ND'].to_numpy(),standard_mumu['ND'].to_numpy()]
        self.NSM = [standard_HHbT['Standard Model'].to_numpy(), standard_HHbV['Standard Model'].to_numpy(),
               standard_LHbT['Standard Model'].to_numpy(), standard_LHbV['Standard Model'].to_numpy(),
              standard_ee['Standard Model'].to_numpy(),standard_mumu['Standard Model'].to_numpy()]


    # calculate branching fraction: supporting functon
    def momentum(self, Mlq, mq, ml):
        a = (Mlq + ml)**2 - mq**2
        b = (Mlq - ml)**2 - mq**2
        return math.sqrt(a*b)/(2*Mlq)

    # calculate branching fraction: supporting functon
    def abseffcoupl_massfactor(self, Mlq, mq, ml):
        return Mlq**2 - (ml**2 + mq**2) - (ml**2 - mq**2)**2/Mlq**2 - (6*ml*mq)

    # calculate branching fraction: supporting functon
    def decay_width_massfactor(self, Mlq, M):
        #M is a list with [Mlq, mq, ml]
        return self.momentum(Mlq,M[0], M[1])*self.abseffcoupl_massfactor(Mlq,M[0],M[1])/(8 * math.pi**2 * Mlq**2)

    # branching fraction main function
    def calculateBranchingFraction(self):
        mev2gev = 0.001
        mass_quarks = {'1': [2.3 , 4.8], '2': [1275, 95], '3': [173070, 4180]}
        mass_leptons= {'1': [0.511, 0.0022], '2': [105.7, 0.17], '3': [1777, 15.5]}

        for gen in mass_quarks:
            mass_quarks[gen] = [x*mev2gev for x in mass_quarks[gen]]

        for gen in mass_leptons:
            mass_leptons[gen] = [x*mev2gev for x in mass_leptons[gen]]

        def make_mass_dict(ls = self.lambdastring):
            md = {}
            for i in range(self.num_lam):
                if ls[i][4] == 'L':
                    md[self.lambdastring[i]] = [[mass_quarks[ls[i][2]][1], mass_leptons[ls[i][3]][0]], [mass_quarks[ls[i][2]][0], mass_leptons[ls[i][3]][1]]]
                elif ls[i][4] == 'R':
                    md[self.lambdastring[i]] = [[mass_quarks[ls[i][2]][1], mass_leptons[ls[i][3]][0]]]
            return md

        mass_dict = make_mass_dict()

        def branching_fraction(ls = self.lambdastring, lm = self.lam, md = mass_dict, Mlq = self.mass, width_const = 0):
            denom = 0
            numer = 0
            for i in range(len(ls)):
                denom += lm[i]**2 * self.decay_width_massfactor(Mlq, mass_dict[ls[i]][0])
                numer += lm[i]**2 * self.decay_width_massfactor(Mlq, mass_dict[ls[i]][0])
                if ls[i][4] == 'L':
                    denom += lm[i]**2 * self.decay_width_massfactor(Mlq, md[ls[i]][1])
            return numer/(denom + width_const)

        self.b_frac = branching_fraction()


    # chi-square calculation function; not called from main
    def get_chisq_ind(self, tag):
        ee_lam = []
        mumu_lam = []
        tautau_lam = []

        for lamda in self.lam:
            temp_str_sym = str(lamda)
            if temp_str_sym[3] == '1':
                ee_lam.append(lamda)
            elif temp_str_sym[3] == '2':
                mumu_lam.append(lamda)
            elif temp_str_sym[3] == '3':
                tautau_lam.append(lamda)
        if (tag<4 and len(self.tautau_eff_l[0])==0) or (tag==4 and len(self.ee_eff_l[0])==0) or (tag==5 and len(self.mumu_eff_l[0])==0):
            return 0
        if tag<4:
            num_bin = len(self.tautau_eff_l[0][0][tag])
        elif tag == 4:
            num_bin = len(self.ee_eff_l[0][0][0])
        elif tag == 5:
            num_bin = len(self.mumu_eff_l[0][0][0])
        nq = [0.0]*num_bin
        np = [0.0]*num_bin
        ns = [0.0]*num_bin
        ni = [0.0]*num_bin
        nt = [0.0]*num_bin
        ntc= [0.0]*num_bin
        nsm= self.NSM[tag]
        nd = self.ND[tag]
        denominator = [nd[bin_no] + 0.01*nd[bin_no]**2 for bin_no in range(num_bin)]
        if tag<4:
            nq = [nq[bin_no] + self.tautau_cs[0][0]*self.tautau_eff_l[0][0][tag][bin_no] * self.b_frac**2 *self.luminosity for bin_no in range(num_bin)]
            for i in range(self.tautau_lambdas_len):
                np = [np[bin_no] + self.tautau_cs[1][i]*self.tautau_eff_l[1][i][tag][bin_no]*tautau_lam[i]**4 *self.b_frac**2 *self.luminosity for bin_no in range(num_bin)]
                ns = [ns[bin_no] + self.tautau_cs[2][i]*self.tautau_eff_l[2][i][tag][bin_no]*tautau_lam[i]**2 *self.b_frac    *self.luminosity for bin_no in range(num_bin)]
                ni = [ni[bin_no] + self.tautau_cs[3][i]*self.tautau_eff_l[3][i][tag][bin_no]*tautau_lam[i]**2               *self.luminosity for bin_no in range(num_bin)]
                nt = [nt[bin_no] + self.tautau_cs[4][i]*self.tautau_eff_l[4][i][tag][bin_no]*tautau_lam[i]**4               *self.luminosity for bin_no in range(num_bin)]
            ntc_cntr = 0
            for i in range(self.tautau_lambdas_len):
                for j in range(i+1, self.tautau_lambdas_len):
                    "use cross-terms"
                    ntc = [ntc[bin_no] + self.cs_tautau_t_ct[ntc_cntr]*self.tautau_eff_t_ct[ntc_cntr][tag][bin_no]* tautau_lam[i]**2 * tautau_lam[j]**2            *self.luminosity for bin_no in range(num_bin)]
                    ntc_cntr += 1
        elif tag==4:
            nq = [nq[bin_no] + self.ee_cs[0][0]*self.ee_eff_l[0][0][0][bin_no] * self.b_frac**2 *self.luminosity for bin_no in range(num_bin)]
            for i in range(self.ee_lambdas_len):
                np = [np[bin_no] + self.ee_cs[1][i]*self.ee_eff_l[1][i][0][bin_no]*ee_lam[i]**4 *self.b_frac**2 *self.luminosity for bin_no in range(num_bin)]
                ns = [ns[bin_no] + self.ee_cs[2][i]*self.ee_eff_l[2][i][0][bin_no]*ee_lam[i]**2 *self.b_frac    *self.luminosity for bin_no in range(num_bin)]
                ni = [ni[bin_no] + self.ee_cs[3][i]*self.ee_eff_l[3][i][0][bin_no]*ee_lam[i]**2               *self.luminosity for bin_no in range(num_bin)]
                nt = [nt[bin_no] + self.ee_cs[4][i]*self.ee_eff_l[4][i][0][bin_no]*ee_lam[i]**4               *self.luminosity for bin_no in range(num_bin)]
            ntc_cntr = 0
            for i in range(self.ee_lambdas_len):
                for j in range(i+1, self.ee_lambdas_len):
                    "use cross-terms"
                    ntc = [ntc[bin_no] + self.cs_ee_t_ct[ntc_cntr]*self.ee_eff_t_ct[ntc_cntr][0][bin_no]* ee_lam[i]**2 * ee_lam[j]**2            *self.luminosity for bin_no in range(num_bin)]
                    ntc_cntr += 1
        elif tag==5:
            nq = [nq[bin_no] + self.mumu_cs[0][0]*self.mumu_eff_l[0][0][0][bin_no] * self.b_frac**2 *self.luminosity for bin_no in range(num_bin)]
            for i in range(self.mumu_lambdas_len):
                np = [np[bin_no] + self.mumu_cs[1][i]*self.mumu_eff_l[1][i][0][bin_no]*mumu_lam[i]**4 *self.b_frac**2 *self.luminosity for bin_no in range(num_bin)]
                ns = [ns[bin_no] + self.mumu_cs[2][i]*self.mumu_eff_l[2][i][0][bin_no]*mumu_lam[i]**2 *self.b_frac    *self.luminosity for bin_no in range(num_bin)]
                ni = [ni[bin_no] + self.mumu_cs[3][i]*self.mumu_eff_l[3][i][0][bin_no]*mumu_lam[i]**2            *self.luminosity for bin_no in range(num_bin)]
                nt = [nt[bin_no] + self.mumu_cs[4][i]*self.mumu_eff_l[4][i][0][bin_no]*mumu_lam[i]**4            *self.luminosity for bin_no in range(num_bin)]
            ntc_cntr = 0
            for i in range(self.mumu_lambdas_len):
                for j in range(i+1, self.mumu_lambdas_len):
                    "use cross-terms"
                    ntc = [ntc[bin_no] + self.cs_mumu_t_ct[ntc_cntr]*self.mumu_eff_t_ct[ntc_cntr][0][bin_no]* mumu_lam[i]**2 * mumu_lam[j]**2            *self.luminosity for bin_no in range(num_bin)]
                    ntc_cntr += 1
        chi_ind = 0.0
        for bin_no in range(num_bin):
            if self.ignorePairSingle:
                chi_ind += ((nq[bin_no]+ni[bin_no]+nt[bin_no]+ntc[bin_no]+nsm[bin_no]-nd[bin_no])**2)/denominator[bin_no] 
            else:
                chi_ind += ((nq[bin_no]+np[bin_no]+ns[bin_no]+ni[bin_no]+nt[bin_no]+ntc[bin_no]+nsm[bin_no]-nd[bin_no])**2)/denominator[bin_no] 
        chi_sq_tag = sym.Add(chi_ind)
        return chi_sq_tag
    
    # calling the chi-square function; not called from main
    def get_chi_square_symb(self):    
        ee_chi = 0
        mumu_chi = 0
        hhbt_chi = 0
        hhbv_chi = 0
        lhbt_chi = 0
        lhbv_chi = 0
        ee_chi = sym.simplify(self.get_chisq_ind(4))
        print("ee done.")
        mumu_chi = sym.simplify(self.get_chisq_ind(5))
        print("mumu done.")
        hhbt_chi = sym.simplify(self.get_chisq_ind(0))
        print("HHbT done.")
        hhbv_chi = sym.simplify(self.get_chisq_ind(1))
        print("HHbV done.")
        lhbt_chi = sym.simplify(self.get_chisq_ind(2))    
        print("LHbT done.")
        lhbv_chi = sym.simplify(self.get_chisq_ind(3))
        print("LHbV done.")
        return sym.Add(ee_chi,mumu_chi,hhbt_chi, hhbv_chi, lhbt_chi, lhbv_chi)

    # converting chi-square sympy expression to numpy
    def getNumpyChiSquare(self):
        self.chisq_symb = sym.simplify(self.get_chi_square_symb())
        self.numpy_chisq=lambdify(flatten(self.lam),self.chisq_symb, modules='numpy')

    # computing the chi-square minima
    def getChiSquareMinima(self):
        print("Computing chi-square minima ... ")
        def f(x):
            z = self.numpy_chisq(*flatten(x))
            return z
        startLambda = 0.5
        startLambdas = np.array([startLambda for x in range(self.num_lam)])
        minima = optimize.minimize(f,startLambdas, method='Nelder-Mead',options={'fatol':0.01})
        self.chisq_min = minima.fun

    # update database
    def updateDatabase(self):
        self.cursor.execute("""INSERT INTO couplings VALUES (?,?,?,?,?)""", (self.mass,self.ignorePairSingle," ".join(self.lambdastring),str(self.chisq_symb),self.chisq_min) )
        self.sqliteConnection.commit() # commit changes
        self.sqliteConnection.close() # close connection

    # printing the output
    def displayResut(self):
        for lam_val,lam_val_copy in zip(self.lam_vals,self.lam_vals_original):
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
            chisq_given_vals = self.numpy_chisq(*flatten(temp))
            if chisq_given_vals - self.chisq_min <= self.chi_sq_limits[len(self.lambdastring)-1]:
                print("Yes")
            else:
                print("No")

    def run(self):
        self.inputValues() # take user input 
        self.convertLambda() # convert lambda type & values to variables & sympy, & sort the lambdas
        self.connectDatabase() # connect to database
        self.checkDatabase() # check if query is already in the database
        cs_ee_t_ct_temp, cs_mumu_t_ct_temp, cs_tautau_t_ct_temp = self.setInterpolatedCrossSection()
        self.setClosestMass()
        self.setEfficiency(cs_ee_t_ct_temp, cs_mumu_t_ct_temp, cs_tautau_t_ct_temp)
        self.readHEPData()
        self.calculateBranchingFraction()
        print("Computing chi-square expression ... ")
        self.getNumpyChiSquare()
        self.getChiSquareMinima()
        self.updateDatabase()
        self.displayResut()

if __name__=="__main__":
    calculator = Calculator()
    calculator.run()