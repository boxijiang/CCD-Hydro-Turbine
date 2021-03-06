import os

basepath = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Desktop', 'RAFTBEM', 'SNOPTDre', '5MW_AFFiles')
os.chdir(basepath)
# C:\Users\boxij\OneDrive\Desktop\RAFTBEM\SNOPTDre\5MW_AFFiles000

import openmdao.api as om
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from ccblade import CCAirfoil, CCBlade


class Computation(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('x', val=np.zeros(nn), desc='velocity', units='rad/s')
        self.add_input('theta', val=np.zeros(7))
        self.add_input('chord', val=np.zeros(7))
        self.add_input('time_phase', units='s', val=np.ones(nn))
        self.add_output('torque', val=np.ones(nn))

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        Rhub = 0.15
        Rtip = 6.3
        nn = self.options['num_nodes']
        r = np.array([0.28667, 0.56000, 0.83333, 1.17500, 1.99500,
                      2.81500, 3.63500, 4.45500, 5.27500, 6.16333])
        thetain = inputs['theta']
        chordin = inputs['chord']
        chordorig = np.array([0.3542, 0.3854, 0.4167])
        thetaorig = np.array([13.308, 13.308, 13.308])
        theta = np.concatenate((thetaorig, thetain), axis=None)
        chord = np.concatenate((chordorig, chordin), axis=None)

        time = inputs['time_phase']
        B = 3  # number of blades

        tilt = 5.0
        precone = 2.5
        yaw = 0.0

        nSector = 8  # azimuthal discretization

        # atmosphere
        rho = 1025
        mu = 0.00109

        # power-law wind shear profile
        shearExp = 0.2
        hubHt = 90.0

        # 1 ----------

        # 2 ----------

        afinit = CCAirfoil.initFromAerodynFile  # just for shorthand

        # load all airfoils
        airfoil_types = [0] * 8
        airfoil_types[0] = afinit('Cylinder1.dat')
        airfoil_types[1] = afinit('Cylinder2.dat')
        airfoil_types[2] = afinit('DU40_A17.dat')
        airfoil_types[3] = afinit('DU35_A17.dat')
        airfoil_types[4] = afinit('DU30_A17.dat')
        airfoil_types[5] = afinit('DU25_A17.dat')
        airfoil_types[6] = afinit('DU21_A17.dat')
        airfoil_types[7] = afinit('NACA64_A17.dat')

        # place at appropriate radial stations
        # af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7] for Nrel
        af_idx = [0, 0, 1, 6, 6, 6, 6, 6, 6, 6] # for simplification
        af = [0] * len(r)
        for i in range(len(r)):
            af[i] = airfoil_types[af_idx[i]]

        # create CCBlade object

        rotor = CCBlade(r, chord, theta, af, Rhub, Rtip, B, rho, mu,
                        precone, tilt, yaw, shearExp, hubHt, nSector)

        # 3 ----------

        # 4 ----------

        # set conditions
        # a = 0.8
        Uinf = np.sin(time) * 0.2 + 1.5
        pitch = np.zeros(nn)
        Omega = inputs['x'] * 30.0 / pi  # convert to RPM
        # Omega = inputs['x'] * 30.0 / pi + 1
        # Omega = inputs['x'] * 30.0 / pi
        azimuth = 0.0

        loads, derivs = rotor.evaluate(Uinf, Omega, pitch)
        outputs['torque'] = loads['Q']
        xx = inputs['x']
        print('x=', xx)






