from quadrotor_agent import quadrotor_agent
from verse import Scenario
from verse.plotter.plotter2D import *
 
import plotly.graph_objects as go
from enum import Enum, auto

import matplotlib.pyplot as plt
from verse.plotter.plotter2D_old import plot_reachtube_tree


class AgentMode(Enum):
    Mode1 = auto() # mode of the uncertain mass (from 0.5m to 3.5m)
    Mode2 = auto() # mode of the nominal mass
    Mode3 = auto() # deterministic mass and 3 times of the nominal value


if __name__ == "__main__":
    input_code_name = './quad_controller.py'
    scenario = Scenario()

    # step 1. create a quadrotor instance with the closed-loop dynamics
    quad = quadrotor_agent('quad1', file_name=input_code_name)
    scenario.add_agent(quad)
    # car = vanderpol_agent('car2', file_name=input_code_name)
    # scenario.add_agent(car)
    # scenario.set_sensor(FakeSensor2())
    # modify mode list input
    # Step 2. change the initial codnitions (set for all 18 states)
    scenario.set_init(
        [[     
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.37, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.63, 0.0]
        ]],
        [
            tuple([AgentMode.Mode1]),
        ]
    )
    t_max = 6
    # traces = scenario.simulate(t_max, 0.01)
    
    # N = int(100*t_max + 1)
    # t = np.linspace(0,t_max,N)
    # x_des_array = []
    # y_des_array = []
    # z_des_array = []

    # for t_step in t:
    #     x_des_array.append(2*(1-np.cos(t_step)))
    #     y_des_array.append(2*np.sin(t_step))
    #     z_des_array.append(1.0 + np.sin(t_step))
    # fig = go.Figure()
    # fig = simulation_tree(traces, None, fig, 0, 1, 'lines', 'trace', print_dim_list=[1,2])
    # fig.add_trace(go.Scatter(x=x_des_array, y=y_des_array,mode="lines",line=dict(color="#0000ff"))) 
    # fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)
    # fig.update_xaxes(range=[-0.5, 4.5],constrain="domain")
    # fig.show()

    traces = scenario.verify(t_max, 0.01)

    fig = plot_reachtube_tree(traces.root, 'quad1', 0, [1])
    plt.show()

    # fig = go.Figure()
    # """use these lines for generating x-y (phase) plots"""
    # fig = reachtube_tree(traces, None, fig, 1, 2,
    #                         'lines', 'trace', print_dim_list=[1,2])
    # fig.add_trace(go.Scatter(x=x_des_array, y=y_des_array,mode="lines",line=dict(color="#0000ff"))) 
    # fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)
    # fig.update_xaxes(range=[-0.5, 4.5],constrain="domain")
    # fig.show()

    # """use these lines for generating x-t (time) plots"""
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 0, 1,
    #                       'lines', 'trace', print_dim_list=[0,1])
    # fig.add_trace(go.Scatter(x=t, y=x_des_array,mode="lines",line=dict(color="#0000ff"))) 
    # fig.update_xaxes(range=[-0.5, 10.5],constrain="domain")
    # fig.show()


    # fig_n = go.Figure()
    # fig_n = reachtube_tree(traces, None, fig_n, 0, 2,
    #                       'lines', 'trace', print_dim_list=[0,2])
    # fig_n.add_trace(go.Scatter(x=t, y=y_des_array,mode="lines",line=dict(color="#0000ff"))) 
    # fig_n.update_xaxes(range=[-0.5, 10.5],constrain="domain")
    # fig_n.show()

    # fig_new = go.Figure()
    # fig_new = reachtube_tree(traces, None, fig_new, 0, 3,
    #                       'lines', 'trace', print_dim_list=[0,3])
    # fig_new.add_trace(go.Scatter(x=t, y=z_des_array,mode="lines",line=dict(color="#0000ff"))) 
    # fig_new.update_xaxes(range=[-0.5, 10.5],constrain="domain")
    # fig_new.show()
