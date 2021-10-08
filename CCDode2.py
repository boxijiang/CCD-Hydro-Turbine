
import os

basepath = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Desktop', 'RAFTBEM', 'SNOPTDre', '5MW_AFFiles')
os.chdir(basepath)
# C:\Users\boxij\OneDrive\Desktop\RAFTBEM\dymosCCBlade\5MW_AFFiles000
import openmdao.api as om
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from ccblade import CCAirfoil, CCBlade
import openmdao.api as om
from BEMcompu2 import Computation


class Turbine(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

    def setup(self):

        nn = self.options['num_nodes']
        # Inputs
        self.add_input('x', val=np.zeros(nn), desc='velocity', units='rad/s')
        # self.add_input('time_phase', val=np.ones(nn), desc='time', units='s')
        self.add_input('u', val=np.ones(nn), desc='control', units=None)
        self.add_input('torque', val=np.ones(nn))
        self.add_input('mass', val=2234)
        self.add_output('xdot', val=np.zeros(nn), desc='velocity component in x', units='rad/s**2',
                         tags=['dymos.state_rate_source:x', 'dymos.state_units:rad/s'])
        self.add_output('Jdot', val=np.ones(nn), desc='derivative of objective', units='1.0/s')
        # Setup partials
        # Complex-step derivatives
        # self.declare_coloring(wrt='*', method='cs', show_sparsity=True)

        self.declare_partials(of='*', wrt='*', method='fd')
    def compute(self, inputs, outputs):

        x = inputs['x']
        u = inputs['u']
        torque = inputs['torque']
        m = inputs['mass']
        outputs['xdot'] = (torque - u) / m
        # outputs['xdot'] = (torque - u) / 3600
        # outputs['xdot'] = (torque - u) / 11776047
        outputs['Jdot'] = torque * x
        Answer = torque * x




class TurbineODE(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('torque_comp', Computation(num_nodes=nn), promotes_inputs=['x'])
        self.add_subsystem('ode_comp', Turbine(num_nodes=nn), promotes_inputs=['u', 'x'], promotes_outputs=['xdot', 'Jdot'])
        self.connect('torque_comp.torque', 'ode_comp.torque')
