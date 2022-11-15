from typing import Tuple, List, Dict
import copy
from dataclasses import dataclass
import numpy as np

from verse.agents.base_agent import BaseAgent
from verse.analysis import Simulator, Verifier, AnalysisTreeNode, AnalysisTree, ReachabilityMethod
from verse.analysis.analysis_tree import AnalysisTreeNodeType
from verse.analysis.utils import sample_rect
from verse.parser.parser import ControllerIR
from verse.sensor.base_sensor import BaseSensor
from verse.map.lane_map import LaneMap

@dataclass
class ScenarioConfig:
    """Configuration for how simulation/verification is performed for a scenario. Properties are
    immutable so that incremental verification works correctly."""

    incremental: bool = False
    """Enable incremental simulation/verification. Results from previous runs will be used to try to
    speed up experiments. Result is undefined when the map, agent dynamics and sensor are changed."""
    unsafe_continue: bool = False
    """Continue exploring the branch when an unsafe condition occurs."""
    init_seg_length: int = 1000
    reachability_method: ReachabilityMethod = ReachabilityMethod.DRYVR
    """Method of performing reachability. Can be DryVR, NeuReach, MixMonoCont and MixMonoDisc."""
    parallel_sim_ahead: int = 8
    """The number of simulation tasks to dispatch before waiting."""
    parallel_ver_ahead: int = 8
    """The number of verification tasks to dispatch before waiting."""
    parallel: bool = True
    """Enable parallelization. Uses the Ray library. Could be slower for small scenarios."""
    try_local: bool = False
    """Heuristic. When enabled, try to use the local thread when some results are cached."""


class Scenario:
    """A simulation/verification scenario."""

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

    def cleanup_cache(self):
        self.past_runs = []
        self.simulator = Simulator(self.config)
        self.verifier = Verifier(self.config)

    def update_config(self, config):
        self.config = config
        self.verifier.config = config
        self.simulator.config = config

    def set_sensor(self, sensor):
        """Sets the sensor for the scenario. Will use the default sensor when not called."""
        self.sensor = sensor

    def set_map(self, track_map: LaneMap):
        """Sets the map for the scenario."""
        self.map = track_map
        # Update the lane mode field in the agent
        for agent_id in self.agent_dict:
            agent = self.agent_dict[agent_id]
            self._update_agent_lane_mode(agent, track_map)

    def add_agent(self, agent: BaseAgent):
        """Adds an agent to the scenario."""
        if self.map is not None:
            # Update the lane mode field in the agent
            self._update_agent_lane_mode(agent, self.map)
        self.agent_dict[agent.id] = agent
        if hasattr(agent, "init_cont") and agent.init_cont is not None:
            self.init_dict[agent.id] = copy.deepcopy(agent.init_cont)
        if hasattr(agent, "init_disc") and agent.init_disc is not None:
            self.init_mode_dict[agent.id] = copy.deepcopy(agent.init_disc)

        if hasattr(agent, "static_parameters") and agent.static_parameters is not None:
            self.static_dict[agent.id] = copy.deepcopy(agent.static_parameters)
        else:
            self.static_dict[agent.id] = []
        if hasattr(agent, "uncertain_parameters") and agent.uncertain_parameters is not None:
            self.uncertain_param_dict[agent.id] = copy.deepcopy(agent.uncertain_parameters)
        else:
            self.uncertain_param_dict[agent.id] = []

    # TODO-PARSER: update this function
    def _update_agent_lane_mode(self, agent: BaseAgent, track_map: LaneMap):
        for lane_id in track_map.lane_dict:
            if (
                "TrackMode" in agent.decision_logic.mode_defs
                and lane_id not in agent.decision_logic.mode_defs["TrackMode"].modes
            ):
                agent.decision_logic.mode_defs["TrackMode"].modes.append(lane_id)
        # mode_vals = list(agent.decision_logic.modes.values())
        # agent.decision_logic.vertices = list(itertools.product(*mode_vals))
        # agent.decision_logic.vertexStrings = [','.join(elem) for elem in agent.decision_logic.vertices]

    def set_init_single(
        self, agent_id, init: list, init_mode: tuple, static=[], uncertain_param=[]
    ):
        """Sets the initial conditions for a single agent."""
        assert agent_id in self.agent_dict, "agent_id not found"
        agent = self.agent_dict[agent_id]
        assert len(init) == 1 or len(init) == 2, "the length of init should be 1 or 2"
        # print(agent.decision_logic.state_defs.values())
        if agent.decision_logic != agent.decision_logic.empty():
            for i in init:
                assert len(i) == len(
                    list(agent.decision_logic.state_defs.values())[0].cont
                ), "the length of element in init not fit the number of continuous variables"
            # print(agent.decision_logic.mode_defs)
            assert len(init_mode) == len(
                list(agent.decision_logic.state_defs.values())[0].disc
            ), "the length of element in init_mode not fit the number of discrete variables"
        if len(init) == 1:
            init = init + init
        self.init_dict[agent_id] = copy.deepcopy(init)
        self.init_mode_dict[agent_id] = copy.deepcopy(init_mode)
        self.agent_dict[agent_id].set_initial(init, init_mode)
        if static:
            self.static_dict[agent_id] = copy.deepcopy(static)
            self.agent_dict[agent_id].set_static_parameter(static)
        else:
            self.static_dict[agent_id] = []
        if uncertain_param:
            self.uncertain_param_dict[agent_id] = copy.deepcopy(uncertain_param)
            self.agent_dict[agent_id].set_uncertain_parameter(uncertain_param)
        else:
            self.uncertain_param_dict[agent_id] = []
        return

    def set_init(self, init_list, init_mode_list, static_list=[], uncertain_param_list=[]):
        """Sets the initial conditions for all agents. The order will be the same as the order in
        which the agents are added."""
        assert len(init_list) == len(
            self.agent_dict
        ), "the length of init_list not fit the number of agents"
        assert len(init_mode_list) == len(
            self.agent_dict
        ), "the length of init_mode_list not fit the number of agents"
        assert (
            len(static_list) == len(self.agent_dict) or len(static_list) == 0
        ), "the length of static_list not fit the number of agents or equal to 0"
        assert (
            len(uncertain_param_list) == len(self.agent_dict) or len(uncertain_param_list) == 0
        ), "the length of uncertain_param_list not fit the number of agents or equal to 0"
        print(init_mode_list)
        print(type(init_mode_list))
        if not static_list:
            static_list = [[] for i in range(0, len(self.agent_dict))]
            # print(static_list)
        if not uncertain_param_list:
            uncertain_param_list = [[] for i in range(0, len(self.agent_dict))]
            # print(uncertain_param_list)
        for i, agent_id in enumerate(self.agent_dict.keys()):
            self.set_init_single(
                agent_id, init_list[i], init_mode_list[i], static_list[i], uncertain_param_list[i]
            )

    def _check_init(self):
        for agent_id in self.agent_dict.keys():
            assert agent_id in self.init_dict, "init of {} not initialized".format(agent_id)
            assert agent_id in self.init_mode_dict, "init_mode of {} not initialized".format(
                agent_id
            )
            assert agent_id in self.static_dict, "static of {} not initialized".format(agent_id)
            assert (
                agent_id in self.uncertain_param_dict
            ), "uncertain_param of {} not initialized".format(agent_id)
        return

    def simulate_multi(self, time_horizon, num_sim):
        res_list = []
        for i in range(num_sim):
            trace = self.simulate(time_horizon)
            res_list.append(trace)
        return res_list

    def simulate(self, time_horizon, time_step, max_height=None, seed=None) -> AnalysisTree:
        """Compute a single simulation trajectory of the scenario, starting from a single initial state.
        `seed`: the random seed for sampling a point in the region specified by the initial
        conditions"""
        _check_ray_init(self.config.parallel)
        self._check_init()
        root = AnalysisTreeNode.root_from_inits(
            init={aid: sample_rect(init, seed) for aid, init in self.init_dict.items()},
            mode={
                aid: tuple(elem if isinstance(elem, str) else elem.name for elem in modes)
                for aid, modes in self.init_mode_dict.items()
            },
            static={aid: [elem.name for elem in modes] for aid, modes in self.static_dict.items()},
            uncertain_param=self.uncertain_param_dict,
            agent=self.agent_dict,
            type=AnalysisTreeNodeType.SIM_TRACE,
            ndigits=10,
        )
        tree = self.simulator.simulate(
            root,
            self.sensor,
            time_horizon,
            time_step,
            max_height,
            self.map,
            len(self.past_runs),
            self.past_runs,
        )
        self.past_runs.append(tree)
        return tree

    def simulate_simple(self, time_horizon, time_step, max_height=None, seed=None) -> AnalysisTree:
        """Compute the set of reachable states, starting from a single point. Evaluates the decision
        logic code directly, and does not use the internal Python parser and generate
        nondeterministic transitions.
        `seed`: the random seed for sampling a point in the region specified by the initial
        conditions"""
        self._check_init()
        root = AnalysisTreeNode.root_from_inits(
            init={aid: sample_rect(init, seed) for aid, init in self.init_dict.items()},
            mode={
                aid: tuple(elem if isinstance(elem, str) else elem.name for elem in modes)
                for aid, modes in self.init_mode_dict.items()
            },
            static={aid: [elem.name for elem in modes] for aid, modes in self.static_dict.items()},
            uncertain_param=self.uncertain_param_dict,
            agent=self.agent_dict,
            type=AnalysisTreeNodeType.SIM_TRACE,
            ndigits=10,
        )
        tree = self.simulator.simulate_simple(
            root,
            time_horizon,
            time_step,
            max_height,
            self.map,
            self.sensor,
            len(self.past_runs),
            self.past_runs,
        )
        self.past_runs.append(tree)
        return tree

    def verify(self, time_horizon, time_step, max_height=None, params={}) -> AnalysisTree:
        """Compute the set of reachable states, starting from a set of initial states states."""
        _check_ray_init(self.config.parallel)
        self._check_init()
        root = AnalysisTreeNode.root_from_inits(
            init={
                aid: [[init, init] if np.array(init).ndim < 2 else init]
                for aid, init in self.init_dict.items()
            },
            mode={
                aid: tuple(elem if isinstance(elem, str) else elem.name for elem in modes)
                for aid, modes in self.init_mode_dict.items()
            },
            static={aid: [elem.name for elem in modes] for aid, modes in self.static_dict.items()},
            uncertain_param=self.uncertain_param_dict,
            agent=self.agent_dict,
            type=AnalysisTreeNodeType.REACH_TUBE,
            ndigits=10,
        )

        tree = self.verifier.compute_full_reachtube(
            root,
            self.sensor,
            time_horizon,
            time_step,
            max_height,
            self.map,
            self.config.init_seg_length,
            self.config.reachability_method,
            len(self.past_runs),
            self.past_runs,
            params,
        )
        self.past_runs.append(tree)
        return tree
