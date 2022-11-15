from pprint import pp
from typing import DefaultDict, NamedTuple, Optional, Tuple, List, Dict, Any
import copy
import itertools
import warnings
from collections import defaultdict, namedtuple
import ast
from dataclasses import dataclass
import types
import sys
from enum import Enum

import numpy as np

from verse.agents.base_agent import BaseAgent
from verse.analysis.dryvr import _EPSILON
from verse.analysis.incremental import CachedRTTrans, CachedSegment, combine_all, reach_trans_suit, sim_trans_suit
from verse.analysis.simulator import PathDiffs
from verse.automaton import GuardExpressionAst, ResetExpression
from verse.analysis import Simulator, Verifier, AnalysisTreeNode, AnalysisTree
from verse.analysis.utils import dedup, sample_rect
from verse.parser import astunparser
from verse.parser.parser import ControllerIR, ModePath, find
from verse.sensor.base_sensor import BaseSensor
from verse.map.lane_map import LaneMap

@dataclass
class ScenarioConfig:
    incremental: bool = False
    unsafe_continue: bool = False
    init_seg_length: int = 1000
    reachability_method: str = 'DRYVR'

class Scenario:
    def __init__(self, config=ScenarioConfig()):
        self.agent_dict: Dict[str, BaseAgent] = {}
        self.simulator = Simulator(config)
        self.verifier = Verifier(config)
        self.init_dict = {}
        self.init_mode_dict = {}
        self.static_dict = {}
        self.uncertain_param_dict = {}
        self.map = LaneMap()
        self.sensor = BaseSensor()
        self.past_runs = []

        # Parameters
        self.config = config

    def set_sensor(self, sensor):
        self.sensor = sensor

    def set_map(self, track_map: LaneMap):
        self.map = track_map
        # Update the lane mode field in the agent
        for agent_id in self.agent_dict:
            agent = self.agent_dict[agent_id]
            self.update_agent_lane_mode(agent, track_map)

    def add_agent(self, agent: BaseAgent):
        if self.map is not None:
            # Update the lane mode field in the agent
            self.update_agent_lane_mode(agent, self.map)
        self.agent_dict[agent.id] = agent
        if hasattr(agent, 'init_cont') and agent.init_cont is not None:
            self.init_dict[agent.id] = copy.deepcopy(agent.init_cont) 
        if hasattr(agent, 'init_disc') and agent.init_disc is not None:
            self.init_mode_dict[agent.id] = copy.deepcopy(agent.init_disc)

        if hasattr(agent, 'static_parameters') and agent.static_parameters is not None:
            self.static_dict[agent.id] = copy.deepcopy(agent.static_parameters)
        else:
            self.static_dict[agent.id] = []
        if hasattr(agent, 'uncertain_parameters') and agent.uncertain_parameters is not None:
            self.uncertain_param_dict[agent.id] = copy.deepcopy(agent.uncertain_parameters)
        else:
            self.uncertain_param_dict[agent.id] = []


    # TODO-PARSER: update this function
    def update_agent_lane_mode(self, agent: BaseAgent, track_map: LaneMap):
        for lane_id in track_map.lane_dict:
            if 'TrackMode' in agent.decision_logic.mode_defs and lane_id not in agent.decision_logic.mode_defs['TrackMode'].modes:
                agent.decision_logic.mode_defs['TrackMode'].modes.append(lane_id)
        # mode_vals = list(agent.decision_logic.modes.values())
        # agent.decision_logic.vertices = list(itertools.product(*mode_vals))
        # agent.decision_logic.vertexStrings = [','.join(elem) for elem in agent.decision_logic.vertices]

    def set_init_single(self, agent_id, init: list, init_mode: tuple, static=[], uncertain_param=[]):
        assert agent_id in self.agent_dict, 'agent_id not found'
        agent = self.agent_dict[agent_id]
        assert len(init) == 1 or len(
            init) == 2, 'the length of init should be 1 or 2'
        # print(agent.decision_logic.state_defs.values())
        if agent.decision_logic != agent.decision_logic.empty():
            for i in init:
                assert len(i) == len(
                    list(agent.decision_logic.state_defs.values())[0].cont),  'the length of element in init not fit the number of continuous variables'
            # print(agent.decision_logic.mode_defs)
            assert len(init_mode) == len(
                list(agent.decision_logic.state_defs.values())[0].disc),  'the length of element in init_mode not fit the number of discrete variables'
        if len(init) == 1:
            init = init+init
        self.init_dict[agent_id] = copy.deepcopy(init)
        self.init_mode_dict[agent_id] = copy.deepcopy(init_mode)
        self.agent_dict[agent_id].set_initial(init, init_mode)
        if static:
            self.static_dict[agent_id] = copy.deepcopy(static)
            self.agent_dict[agent_id].set_static_parameter(static)
        else:
            self.static_dict[agent_id] = []
        if uncertain_param:
            self.uncertain_param_dict[agent_id] = copy.deepcopy(
                uncertain_param)
            self.agent_dict[agent_id].set_uncertain_parameter(uncertain_param)
        else:
            self.uncertain_param_dict[agent_id] = []
        return

    def set_init(self, init_list, init_mode_list, static_list=[], uncertain_param_list=[]):
        assert len(init_list) == len(
            self.agent_dict), 'the length of init_list not fit the number of agents'
        assert len(init_mode_list) == len(
            self.agent_dict), 'the length of init_mode_list not fit the number of agents'
        assert len(static_list) == len(
            self.agent_dict) or len(static_list) == 0, 'the length of static_list not fit the number of agents or equal to 0'
        assert len(uncertain_param_list) == len(self.agent_dict)\
            or len(uncertain_param_list) == 0, 'the length of uncertain_param_list not fit the number of agents or equal to 0'
        print(init_mode_list)
        print(type(init_mode_list))
        if not static_list:
            static_list = [[] for i in range(0, len(self.agent_dict))]
            # print(static_list)
        if not uncertain_param_list:
            uncertain_param_list = [[] for i in range(0, len(self.agent_dict))]
            # print(uncertain_param_list)
        for i, agent_id in enumerate(self.agent_dict.keys()):
            self.set_init_single(agent_id, init_list[i],
                                 init_mode_list[i], static_list[i], uncertain_param_list[i])

    def check_init(self):
        for agent_id in self.agent_dict.keys():
            assert agent_id in self.init_dict, 'init of {} not initialized'.format(
                agent_id)
            assert agent_id in self.init_mode_dict, 'init_mode of {} not initialized'.format(
                agent_id)
            assert agent_id in self.static_dict, 'static of {} not initialized'.format(
                agent_id)
            assert agent_id in self.uncertain_param_dict, 'uncertain_param of {} not initialized'.format(
                agent_id)
        return

    def simulate_multi(self, time_horizon, num_sim):
        res_list = []
        for i in range(num_sim):
            trace = self.simulate(time_horizon)
            res_list.append(trace)
        return res_list

    def simulate(self, time_horizon, time_step, seed = None) -> AnalysisTree:
        self.check_init()
        init_list = []
        init_mode_list = []
        static_list = []
        agent_list = []
        uncertain_param_list = []
        for agent_id in self.agent_dict:
            init_list.append(sample_rect(self.init_dict[agent_id], seed))
            init_mode_list.append(self.init_mode_dict[agent_id])
            static_list.append(self.static_dict[agent_id])
            uncertain_param_list.append(self.uncertain_param_dict[agent_id])
            agent_list.append(self.agent_dict[agent_id])
        print(init_list)
        tree = self.simulator.simulate(init_list, init_mode_list, static_list, uncertain_param_list, agent_list, self, time_horizon, time_step, self.map, len(self.past_runs), self.past_runs)
        self.past_runs.append(tree)
        return tree

    def simulate_simple(self, time_horizon, time_step, seed = None) -> AnalysisTree:
        self.check_init()
        init_list = []
        init_mode_list = []
        static_list = []
        agent_list = []
        uncertain_param_list = []
        for agent_id in self.agent_dict:
            init_list.append(sample_rect(self.init_dict[agent_id], seed))
            init_mode_list.append(self.init_mode_dict[agent_id])
            static_list.append(self.static_dict[agent_id])
            uncertain_param_list.append(self.uncertain_param_dict[agent_id])
            agent_list.append(self.agent_dict[agent_id])
        print(init_list)
        tree = self.simulator.simulate_simple(init_list, init_mode_list, static_list, uncertain_param_list, agent_list, self, time_horizon, time_step, self.map, len(self.past_runs), self.past_runs)
        self.past_runs.append(tree)
        return tree

    def verify(self, time_horizon, time_step, params={}) -> AnalysisTree:
        self.check_init()
        init_list = []
        init_mode_list = []
        static_list = []
        agent_list = []
        uncertain_param_list = []
        for agent_id in self.agent_dict:
            init = self.init_dict[agent_id]
            tmp = np.array(init)
            if tmp.ndim < 2:
                init = [init, init]
            init_list.append(init)
            init_mode_list.append(self.init_mode_dict[agent_id])
            static_list.append(self.static_dict[agent_id])
            uncertain_param_list.append(self.uncertain_param_dict[agent_id])
            agent_list.append(self.agent_dict[agent_id])
        tree = self.verifier.compute_full_reachtube(init_list, init_mode_list, static_list, uncertain_param_list, agent_list, self, time_horizon,
                                                    time_step, self.map, self.config.init_seg_length, self.config.reachability_method, len(self.past_runs), self.past_runs, params)
        self.past_runs.append(tree)
        return tree
