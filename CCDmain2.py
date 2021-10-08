import openmdao.api as om
import dymos as dm
import numpy as np
from CCDode2 import TurbineODE
import matplotlib.pyplot as plt
import time

prob = om.Problem()
prob.model = om.Group()
prob.driver = om.pyOptSparseDriver()
prob.driver.options['optimizer'] = 'SNOPT'
# prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)


prob.driver.opt_settings['Major iterations limit'] = 1000
prob.driver.opt_settings['Major feasibility tolerance'] = 1e-8
recorder = om.SqliteRecorder('/Users/boxij/OneDrive/Desktop/RAFTBEM/SNOPTDre/OPTChydro/cases.sql')
prob.add_recorder(recorder)
prob.driver.add_recorder(recorder)

A = time.time()
num_seg = 12
traj = prob.model.add_subsystem('traj', dm.Trajectory())

phase = traj.add_phase('phase0',
                       dm.Phase(ode_class=TurbineODE,
                                transcription=dm.GaussLobatto(num_segments=num_seg, compressed=True)))
phase.add_state('x', fix_initial=False, lower=0,
                rate_source='xdot',
                units=None,
                targets='x')  # target required because x0 is an input
phase.add_state('J', fix_initial=True, fix_final=False,
                rate_source='Jdot',
                units=None, ref=1e6)

# phase.add_control(name='u', units=None, continuity=True, rate_continuity=True, targets='u')
phase.add_control(name='u', units=None, lower=0, upper=80000, continuity=True,
                  rate_continuity=True, targets='u', ref=1e5)

# NREL Initial
 # CCD
phase.add_parameter('theta', targets=['torque_comp.theta'], static_target=True, units=None,
                    val=np.array([13.308, 10.162, 7.795, 5.361, 3.125, 1.526, 0.106]),
                    opt=True, lower=np.zeros(7), upper=30 * np.ones(7))
phase.add_parameter('chord', targets=['torque_comp.chord'], static_target=True, units=None,
                    val=np.array([0.4557, 0.4458, 0.4007, 0.3502, 0.301, 0.2518, 0.1419]),
                    opt=True, lower=np.zeros(7), upper=np.ones(7))

''' # Optimal Control
phase.add_parameter('theta', targets=['torque_comp.theta'], static_target=True, units=None,
                    val=np.array([13.308, 10.162, 7.795, 5.361, 3.125, 1.526, 0.106]),
                    opt=False)
phase.add_parameter('chord', targets=['torque_comp.chord'], static_target=True, units=None,
                    val=np.array([0.4557, 0.4458, 0.4007, 0.3502, 0.301, 0.2518, 0.1419]),
                    opt=False)
'''

phase.add_parameter('mass', targets=['ode_comp.mass'], static_target=True, units=None,
                    val=2234, opt=False)
t_final = 20
phase.set_time_options(fix_initial=True, fix_duration=True, duration_val=t_final, units='s', targets='torque_comp.time_phase')

phase.add_timeseries_output('torque_comp.torque')

phase.add_objective('J', loc='final', ref=-1.0e6)

prob.setup()

# phase.add_parameter('theta', targets=['theta'], units=None, opt=True)

prob['traj.phase0.t_initial'] = 0.0
prob['traj.phase0.t_duration'] = t_final

prob['traj.phase0.states:x'] = phase.interp(ys=[2, 2], nodes='state_input')
prob['traj.phase0.states:J'] = phase.interp(ys=[0, 1000000], nodes='state_input')
prob['traj.phase0.controls:u'] = phase.interp(ys=[50000, 50000], nodes='control_input')
prob['traj.phase0.parameters:theta'] = TWTWTW = np.array([13.308, 10.162, 7.795, 5.361, 3.125, 1.526, 0.106])
prob['traj.phase0.parameters:chord'] = CCCCC = np.array([0.4557, 0.4458, 0.4007, 0.3502, 0.301, 0.2518, 0.1419])

# prob.model.traj.phases.phase0.set_refine_options(refine=True)
# dm.run_problem(prob, refine_iteration_limit=1)
dm.run_problem(prob)
J = prob.compute_totals(driver_scaling=True, return_format='flat_dict')
prob.set_solver_print(0)
prob.record("final_state")


# prob.run_model()
B = time.time()
C = B - A
print('Computime=', C)


JJJ = prob.get_val('traj.phase0.timeseries.states:J')
TTT = prob.get_val('traj.phase0.timeseries.time')
XXX = prob.get_val('traj.phase0.timeseries.states:x')
UUU = prob.get_val('traj.phase0.timeseries.controls:u')
TTTORQUE = prob.get_val('traj.phase0.timeseries.torque')

twist = prob.get_val('traj.phase0.timeseries.parameters:theta')
chord = prob.get_val('traj.phase0.timeseries.parameters:chord')
twist_final = twist[0]
print('Twistfinal=', twist_final)
chord_final = chord[0]
print('chordfinal=', chord_final)

twist_original = TWTWTW
chord_original = CCCCC
r = np.array([1.17500, 1.99500, 2.81500, 3.63500, 4.45500, 5.27500, 6.16333])
fig = plt.figure()
cmap = plt.get_cmap("tab10")
plt.plot(r, twist_original, color=cmap(2), marker="o", label='initial')
plt.plot(r, twist_final, color=cmap(3), marker="o", label='final')
plt.legend(loc='upper right')
plt.xlabel('Position along radius (m)')
plt.ylabel('Twist angle along radius')
plt.savefig('/Users/boxij/OneDrive/Desktop/RAFTBEM/SNOPTDre/OPTChydro/TWISTRAMPCCD.png')

fig = plt.figure()
cmap = plt.get_cmap("tab10")
plt.plot(r, chord_original, color=cmap(2), marker="o", label='initial')
plt.plot(r, chord_final, color=cmap(3), marker="o", label='final')
plt.legend(loc='upper right')
plt.xlabel('Position along radius (m)')
plt.ylabel('Chord length along radius')
plt.savefig('/Users/boxij/OneDrive/Desktop/RAFTBEM/SNOPTDre/OPTChydro/CHORDRAMPCCD.png')

fig = plt.figure()
cmap = plt.get_cmap("tab10")
# plt.plot(TT, JJ, color=cmap(0), marker="o")
plt.plot(TTT, JJJ, color=cmap(1), marker="o")
plt.xlabel('Time (s)')
plt.ylabel('Output Energy')
plt.savefig('/Users/boxij/OneDrive/Desktop/RAFTBEM/SNOPTDre/OPTChydro/OutputEnergyRAMPCCD.png')

fig2 = plt.figure()
cmap = plt.get_cmap("tab10")
# plt.plot(TT, XX, color=cmap(0), marker="o")
plt.plot(TTT, XXX, color=cmap(1), marker="o")
plt.xlabel('Time (s)')
plt.ylabel('Turbine rotating speed (rad/s)')
plt.savefig('/Users/boxij/OneDrive/Desktop/RAFTBEM/SNOPTDre/OPTChydro/TURBINESPEEDRAMPCCD.png')

plt.figure()
cmap = plt.get_cmap("tab10")
# plt.plot(TT, UU, color=cmap(0), marker="o")
plt.plot(TTT, UUU, color=cmap(1), marker="o")
plt.xlabel('Time (s)')
plt.ylabel('Control Force')
plt.savefig('/Users/boxij/OneDrive/Desktop/RAFTBEM/SNOPTDre/OPTChydro/CONTROLLOADRAMPCCD.png')

sim_out = traj.simulate()
JJ = sim_out.get_val('traj.phase0.timeseries.states:J')
TT = sim_out.get_val('traj.phase0.timeseries.time')
XX = sim_out.get_val('traj.phase0.timeseries.states:x')
UU = sim_out.get_val('traj.phase0.timeseries.controls:u')
TTORQUE = sim_out.get_val('traj.phase0.timeseries.torque')
print('Time', np.transpose(TT))

file = open("/Users/boxij/OneDrive/Desktop/RAFTBEM/SNOPTDre/OPTChydro/dataramp.txt", "w")

str = repr(C)
file.write("COMPUTime = " + str + "\n")

str = repr(np.transpose(TT))
file.write("Time = " + str + "\n")

str = repr(np.transpose(TTT))
file.write("Time = " + str + "\n")

str = repr(np.transpose(JJ))
file.write("Energy = " + str + "\n")

str = repr(np.transpose(JJJ))
file.write("Energy = " + str + "\n")

str = repr(np.transpose(XX))
file.write("Turbine speed = " + str + "\n")

str = repr(np.transpose(XXX))
file.write("Turbine speed = " + str + "\n")

str = repr(np.transpose(UU))
file.write("Control = " + str + "\n")

str = repr(np.transpose(UUU))
file.write("Control = " + str + "\n")

str = repr(np.transpose(TTORQUE))
file.write("Fluid torque = " + str + "\n")

str = repr(np.transpose(TTTORQUE))
file.write("Fluid torque = " + str + "\n")

str = repr(twist[0])
file.write("twist_final = " + str + "\n")

str = repr(TWTWTW)
file.write("twist_initial = " + str + "\n")

str = repr(chord[0])
file.write("chord_final = " + str + "\n")

file = open("/Users/boxij/OneDrive/Desktop/RAFTBEM/SNOPTDre/OPTChydro/dataramp.txt", "w")

str = repr(C)
file.write("COMPUTime = " + str + "\n")

str = repr(np.transpose(TT))
file.write("Time = " + str + "\n")

str = repr(np.transpose(TTT))
file.write("Time = " + str + "\n")

str = repr(np.transpose(JJ))
file.write("Energy = " + str + "\n")

str = repr(np.transpose(JJJ))
file.write("Energy = " + str + "\n")

str = repr(np.transpose(XX))
file.write("Turbine speed = " + str + "\n")

str = repr(np.transpose(XXX))
file.write("Turbine speed = " + str + "\n")

str = repr(np.transpose(UU))
file.write("Control = " + str + "\n")

str = repr(np.transpose(UUU))
file.write("Control = " + str + "\n")

str = repr(np.transpose(TTORQUE))
file.write("Fluid torque = " + str + "\n")

str = repr(np.transpose(TTTORQUE))
file.write("Fluid torque = " + str + "\n")

str = repr(twist[0])
file.write("twist_final = " + str + "\n")

str = repr(TWTWTW)
file.write("twist_initial = " + str + "\n")

str = repr(chord[0])
file.write("chord_final = " + str + "\n")

str = repr(CCCCC)
file.write("chord_initial = " + str + "\n")

file.close()

#############################################################################



fig = plt.figure()
cmap = plt.get_cmap("tab10")
plt.plot(TT, JJ, color=cmap(0), marker="o")
plt.plot(TTT, JJJ, color=cmap(1), marker="o")
plt.xlabel('Time (s)')
plt.ylabel('Output Energy')
plt.savefig('/Users/boxij/OneDrive/Desktop/RAFTBEM/SNOPTDre/OPTChydro/OutputEnergyRAMPCCD2.png')

fig2 = plt.figure()
cmap = plt.get_cmap("tab10")
plt.plot(TT, XX, color=cmap(0), marker="o")
plt.plot(TTT, XXX, color=cmap(1), marker="o")
plt.xlabel('Time (s)')
plt.ylabel('Turbine rotating speed (rad/s)')
plt.savefig('/Users/boxij/OneDrive/Desktop/RAFTBEM/SNOPTDre/OPTChydro/TURBINESPEEDRAMPCCD2.png')

plt.figure()
cmap = plt.get_cmap("tab10")
plt.plot(TT, UU, color=cmap(0), marker="o")
plt.plot(TTT, UUU, color=cmap(1), marker="o")
plt.xlabel('Time (s)')
plt.ylabel('Control Force')
plt.savefig('/Users/boxij/OneDrive/Desktop/RAFTBEM/SNOPTDre/OPTChydro/CONTROLLOADRAMPCCD2.png')

