import functools
import itertools
import pprint
from typing import List, Dict, Tuple
import copy
from collections import defaultdict, namedtuple
import warnings
import ast

import numpy as np

# from verse.agents.base_agent import BaseAgent
from verse.analysis.analysis_tree import AnalysisTreeNode, AnalysisTree
from verse.analysis.dryvr import calc_bloated_tube_dryvr, SIMTRACENUM
from verse.analysis.mixmonotone import calculate_bloated_tube_mixmono_cont, calculate_bloated_tube_mixmono_disc
from verse.analysis.incremental import ReachTubeCache, TubeCache, convert_reach_trans, to_simulate, combine_all
from verse.analysis.utils import dedup
from verse.parser.parser import find
from verse.analysis.incremental import CachedRTTrans, CachedSegment, combine_all, reach_trans_suit, sim_trans_suit
from verse.automaton import GuardExpressionAst, ResetExpression
from verse.agents.base_agent import BaseAgent
from verse.parser.parser import ControllerIR, ModePath, find
from verse.analysis.utils import EGO, OTHERS

pp = functools.partial(pprint.pprint, compact=True, width=130)

class Verifier:
    def __init__(self, config):
        self.reachtube_tree = None
        self.cache = TubeCache()
        self.trans_cache = ReachTubeCache()
        self.tube_cache_hits = (0, 0)
        self.trans_cache_hits = (0, 0)
        self.config = config

    def calculate_full_bloated_tube(
        self,
        agent_id,
        mode_label,
        initial_set,
        time_horizon,
        time_step,
        sim_func,
        params,
        kvalue,
        sim_trace_num,
        combine_seg_length = 1000,
        guard_checker=None,
        guard_str="",
        lane_map = None
    ):
        # Handle Parameters
        bloating_method = 'PW'
        if 'bloating_method' in params:
            bloating_method = params['bloating_method']
        
        res_tube = None
        tube_length = 0
        if len(initial_set)> combine_seg_length:
            print('stop')
        for combine_seg_idx in range(0, len(initial_set), combine_seg_length):
            rect_seg = initial_set[combine_seg_idx:combine_seg_idx+combine_seg_length]
            combined_rect = None
            for rect in rect_seg:
                rect = np.array(rect)
                if combined_rect is None:
                    combined_rect = rect
                else:
                    combined_rect[0, :] = np.minimum(
                        combined_rect[0, :], rect[0, :])
                    combined_rect[1, :] = np.maximum(
                        combined_rect[1, :], rect[1, :])
            combined_rect = combined_rect.tolist()
            if self.config.incremental:
                cached = self.cache.check_hit(agent_id, mode_label, combined_rect)
                if cached != None:
                    self.tube_cache_hits = self.tube_cache_hits[0] + 1, self.tube_cache_hits[1]
                else:
                    self.tube_cache_hits = self.tube_cache_hits[0], self.tube_cache_hits[1] + 1
            else:
                cached = None
            if cached != None:
                cur_bloated_tube = cached.tube
            else:
                cur_bloated_tube = calc_bloated_tube_dryvr(mode_label,
                                            combined_rect,
                                            time_horizon,
                                            time_step, 
                                            sim_func,
                                            bloating_method,
                                            kvalue,
                                            sim_trace_num,
                                            lane_map = lane_map
                                            )
                if self.config.incremental:
                    self.cache.add_tube(agent_id, mode_label, combined_rect, cur_bloated_tube)
            if combine_seg_idx == 0:
                res_tube = cur_bloated_tube
                tube_length = cur_bloated_tube.shape[0]
            else:
                cur_bloated_tube = cur_bloated_tube[:tube_length - combine_seg_idx*2,:]
                # Handle Lower Bound
                res_tube[combine_seg_idx*2::2,1:] = np.minimum(
                    res_tube[combine_seg_idx*2::2,1:],
                    cur_bloated_tube[::2,1:]
                )
                # Handle Upper Bound
                res_tube[combine_seg_idx*2+1::2,1:] = np.maximum(
                    res_tube[combine_seg_idx*2+1::2,1:],
                    cur_bloated_tube[1::2,1:]
                )
        return res_tube.tolist()

    def apply_cont_var_updater(self, cont_var_dict, updater):
        for variable in updater:
            for unrolled_variable, unrolled_variable_index in updater[variable]:
                cont_var_dict[unrolled_variable] = cont_var_dict[variable][unrolled_variable_index]

    def _get_combinations(self, symbols, cont_var_dict):
        data_list = []
        for symbol in symbols:
            data_list.append(cont_var_dict[symbol])
        comb_list = list(itertools.product(*data_list))
        return comb_list

    def apply_reset(self, agent: BaseAgent, reset_list, all_agent_state, track_map) -> Tuple[str, np.ndarray]:
        dest = []
        rect = []

        agent_state, agent_mode, agent_static = all_agent_state[agent.id]

        dest = copy.deepcopy(agent_mode)
        possible_dest = [[elem] for elem in dest]
        ego_type = find(agent.decision_logic.args, lambda a: a.name == EGO).typ
        rect = copy.deepcopy([agent_state[0][1:], agent_state[1][1:]])

        # The reset_list here are all the resets for a single transition. Need to evaluate each of them
        # and then combine them together
        for reset_tuple in reset_list:
            reset, disc_var_dict, cont_var_dict, _, _p = reset_tuple
            reset_variable = reset.var
            expr = reset.expr
            # First get the transition destinations
            if "mode" in reset_variable:
                found = False
                for var_loc, discrete_variable_ego in enumerate(agent.decision_logic.state_defs[ego_type].disc):
                    if discrete_variable_ego == reset_variable:
                        found = True
                        break
                if not found:
                    raise ValueError(
                        f'Reset discrete variable {discrete_variable_ego} not found')
                if isinstance(reset.val_ast, ast.Constant):
                    val = eval(expr)
                    possible_dest[var_loc] = [val]
                else:
                    tmp = expr.split('.')
                    if 'map' in tmp[0]:
                        for var in disc_var_dict:
                            expr = expr.replace(var, f"'{disc_var_dict[var]}'")
                        res = eval(expr)
                        if not isinstance(res, list):
                            res = [res]
                        possible_dest[var_loc] = res
                    else:
                        expr = tmp
                        if expr[0].strip(' ') in agent.decision_logic.mode_defs:
                            possible_dest[var_loc] = [expr[1]]

            # Assume linear function for continuous variables
            else:
                lhs = reset_variable
                rhs = expr
                found = False
                for lhs_idx, cts_variable in enumerate(agent.decision_logic.state_defs[ego_type].cont):
                    if cts_variable == lhs:
                        found = True
                        break
                if not found:
                    raise ValueError(
                        f'Reset continuous variable {cts_variable} not found')
                # substituting low variables

                symbols = []
                for var in cont_var_dict:
                    if var in expr:
                        symbols.append(var)

                # TODO: Implement this function
                # The input to this function is a list of used symbols and the cont_var_dict
                # The ouput of this function is a list of tuple of values for each variable in the symbols list
                # The function will explor all possible combinations of low bound and upper bound for the variables in the symbols list
                comb_list = self._get_combinations(symbols, cont_var_dict)

                lb = float('inf')
                ub = -float('inf')

                for comb in comb_list:
                    val_dict = {}
                    tmp = copy.deepcopy(expr)
                    for symbol_idx, symbol in enumerate(symbols):
                        tmp = tmp.replace(symbol, str(comb[symbol_idx]))
                    res = eval(tmp, {}, val_dict)
                    lb = min(lb, res)
                    ub = max(ub, res)

                rect[0][lhs_idx] = lb
                rect[1][lhs_idx] = ub

        all_dest = itertools.product(*possible_dest)
        dest = []
        for tmp in all_dest:
            dest.append(tmp)

        return dest, rect

    def postCont(
        self, 
        node: AnalysisTreeNode,
        remain_time: float,
        time_step: float, 
        lane_map, 
        combine_seg_length: int = 1000, 
        reachability_method: str = 'DRYVR', 
        params: Dict = {}
    ):
        for agent_id in node.agent:
            mode = node.mode[agent_id]
            inits = node.init[agent_id]
            if agent_id not in node.trace:
                initial_set = inits 
                res_tube = None 
                tube_length = 0
                for combine_seg_idx in range(0, len(initial_set), combine_seg_length):
                    rect_seg = initial_set[combine_seg_idx:combine_seg_idx+combine_seg_length]
                    combined_rect = None
                    for rect in rect_seg:
                        rect = np.array(rect)
                        if combined_rect is None:
                            combined_rect = rect
                        else:
                            combined_rect[0, :] = np.minimum(
                                combined_rect[0, :], rect[0, :])
                            combined_rect[1, :] = np.maximum(
                                combined_rect[1, :], rect[1, :])
                    combined_rect = combined_rect.tolist()
                

                    # Compute the trace starting from initial condition
                    if reachability_method == "DRYVR":
                        from verse.analysis.dryvr import calc_bloated_tube_dryvr, SIMTRACENUM
                        # pp(('tube', agent_id, mode, inits))
                        bloating_method = 'PW'
                        if 'bloating_method' in params:
                            bloating_method = params['bloating_method']
                        
                        res_tube = calc_bloated_tube_dryvr(
                                            mode,
                                            combined_rect,
                                            remain_time,
                                            time_step, 
                                            node.agent[agent_id].TC_simulate,
                                            bloating_method,
                                            100,
                                            SIMTRACENUM,
                                            lane_map = lane_map
                                            )
                    elif reachability_method == "NeuReach":
                        from verse.analysis.NeuReach.NeuReach_onestep_rect import calculate_bloated_tube_NeuReach
                        res_tube = calculate_bloated_tube_NeuReach(
                            mode, 
                            combined_rect, 
                            remain_time, 
                            time_step, 
                            node.agent[agent_id].TC_simulate, 
                            lane_map,
                            params, 
                        )
                    elif reachability_method == "MIXMONO_CONT":
                        uncertain_param = node.uncertain_param[agent_id]
                        res_tube = calculate_bloated_tube_mixmono_cont(
                            mode, 
                            combined_rect, 
                            uncertain_param, 
                            remain_time,
                            time_step, 
                            node.agent[agent_id],
                            lane_map
                        )
                    elif reachability_method == "MIXMONO_DISC":
                        uncertain_param = node.uncertain_param[agent_id]
                        res_tube = calculate_bloated_tube_mixmono_disc(
                            mode, 
                            combined_rect, 
                            uncertain_param,
                            remain_time,
                            time_step,
                            node.agent[agent_id],
                            lane_map
                        ) 
                    else:
                        raise ValueError(f"Reachability computation method {reachability_method} not available.")
                
                    if combine_seg_idx == 0:
                        cur_bloated_tube = res_tube 
                        tube_length = cur_bloated_tube.shape[0]
                    else:
                        res_tube = res_tube[:tube_length-combine_seg_idx*2,:]
                        # Handle Lower Bound
                        cur_bloated_tube[combine_seg_idx*2::2,1:] = np.minimum(
                            cur_bloated_tube[combine_seg_idx*2::2,1:],
                            res_tube[::2,1:]
                        )
                        # Handle Upper Bound
                        cur_bloated_tube[combine_seg_idx*2+1::2,1:] = np.maximum(
                            cur_bloated_tube[combine_seg_idx*2+1::2,1:],
                            res_tube[1::2,1:]
                        )
                
                cur_bloated_tube[:, 0] += node.start_time
                node.trace[agent_id] = cur_bloated_tube.tolist()
        return node

    def postDisc(
        self,
        node: AnalysisTreeNode,
        track_map, 
        sensor,
        cache: Dict={},
        paths=[],

    ):
        # For each agent
        agent_guard_dict = defaultdict(list)
        cached_guards = defaultdict(list)
        min_trans_ind = None
        cached_trans = defaultdict(list)

        if not cache:
            paths = [(agent, p) for agent in node.agent.values() for p in agent.decision_logic.paths]
        else:
            # _transitions = [trans.transition for seg in cache.values() for trans in seg.transitions]
            _transitions = [(aid, trans) for aid, seg in cache.items() for trans in seg.transitions if reach_trans_suit(trans.inits, node.init)]
            # pp(("cached trans", len(_transitions)))
            if len(_transitions) > 0:
                min_trans_ind = min([t.transition for _, t in _transitions])
                # TODO: check for asserts
                cached_trans = [(aid, tran.mode, tran.dest, tran.reset, tran.reset_idx, tran.paths) for aid, tran in dedup(_transitions, lambda p: (p[0], p[1].mode, p[1].dest)) if tran.transition == min_trans_ind]
                if len(paths) == 0:
                    # print(red("full cache"))
                    return None, cached_trans

                path_transitions = defaultdict(int)
                for seg in cache.values():
                    for tran in seg.transitions:
                        for p in tran.paths:
                            path_transitions[p.cond] = max(path_transitions[p.cond], tran.transition)
                for agent_id, segment in cache.items():
                    agent = node.agent[agent_id]
                    if len(agent.decision_logic.args) == 0:
                        continue
                    state_dict = {aid: (node.trace[aid][0], node.mode[aid], node.static[aid]) for aid in node.agent}

                    agent_paths = dedup([p for tran in segment.transitions for p in tran.paths], lambda i: (i.var, i.cond, i.val))
                    for path in agent_paths:
                        cont_var_dict_template, discrete_variable_dict, length_dict = sensor.sense(
                            self, agent, state_dict, track_map)
                        reset = (path.var, path.val_veri)
                        guard_expression = GuardExpressionAst([path.cond_veri])

                        cont_var_updater = guard_expression.parse_any_all_new(
                            cont_var_dict_template, discrete_variable_dict, length_dict)
                        self.apply_cont_var_updater(
                            cont_var_dict_template, cont_var_updater)
                        guard_can_satisfied = guard_expression.evaluate_guard_disc(
                            agent, discrete_variable_dict, cont_var_dict_template, track_map)
                        if not guard_can_satisfied:
                            continue
                        cached_guards[agent_id].append((path, guard_expression, cont_var_updater, copy.deepcopy(discrete_variable_dict), reset, path_transitions[path.cond]))

        # for aid, trace in node.trace.items():
        #     if len(trace) < 2:
        #         pp(("weird state", aid, trace))
        for agent, path in paths:
            if len(agent.decision_logic.args) == 0:
                continue
            agent_id = agent.id
            state_dict = {aid: (node.trace[aid][0:2], node.mode[aid], node.static[aid]) for aid in node.agent}
            cont_var_dict_template, discrete_variable_dict, length_dict = sensor.sense(
                self, agent, state_dict, track_map)
            # TODO-PARSER: Get equivalent for this function
            # Construct the guard expression
            guard_expression = GuardExpressionAst([path.cond_veri])

            cont_var_updater = guard_expression.parse_any_all_new(
                cont_var_dict_template, discrete_variable_dict, length_dict)
            self.apply_cont_var_updater(
                cont_var_dict_template, cont_var_updater)
            guard_can_satisfied = guard_expression.evaluate_guard_disc(
                agent, discrete_variable_dict, cont_var_dict_template, track_map)
            if not guard_can_satisfied:
                continue
            agent_guard_dict[agent_id].append(
                (guard_expression, cont_var_updater, copy.deepcopy(discrete_variable_dict), path))

        trace_length = int(min(len(v) for v in node.trace.values()) // 2)
        # pp(("trace len", trace_length, {a: len(t) for a, t in node.trace.items()}))
        guard_hits = []
        guard_hit = False
        for idx in range(trace_length):
            if idx == 1500:
                print("stop")
            if min_trans_ind != None and idx >= min_trans_ind:
                return None, cached_trans
            any_contained = False
            hits = []
            state_dict = {aid: (node.trace[aid][idx*2:idx*2+2], node.mode[aid], node.static[aid]) for aid in node.agent}

            asserts = defaultdict(list)
            for agent_id in node.agent.keys():
                agent: BaseAgent = node.agent[agent_id]
                if len(agent.decision_logic.args) == 0:
                    continue
                agent_state, agent_mode, agent_static = state_dict[agent_id]
                # if np.array(agent_state).ndim != 2:
                #     pp(("weird state", agent_id, agent_state))
                agent_state = agent_state[1:]
                cont_vars, disc_vars, len_dict = sensor.sense(self, agent, state_dict, track_map)
                resets = defaultdict(list)
                # Check safety conditions
                for i, a in enumerate(agent.decision_logic.asserts_veri):
                    pre_expr = a.pre

                    def eval_expr(expr):
                        ge = GuardExpressionAst([copy.deepcopy(expr)])
                        cont_var_updater = ge.parse_any_all_new(cont_vars, disc_vars, len_dict)
                        self.apply_cont_var_updater(cont_vars, cont_var_updater)
                        sat = ge.evaluate_guard_disc(agent, disc_vars, cont_vars, track_map)
                        if sat:
                            sat = ge.evaluate_guard_hybrid(agent, disc_vars, cont_vars, track_map)
                            if sat:
                                sat, contained = ge.evaluate_guard_cont(agent, cont_vars, track_map)
                                sat = sat and contained
                        return sat
                    if eval_expr(pre_expr):
                        if not eval_expr(a.cond):
                            label = a.label if a.label != None else f"<assert {i}>"
                            print(f"assert hit for {agent_id}: \"{label}\"")
                            print(idx)
                            asserts[agent_id].append(label)
                if agent_id in asserts:
                    continue
                if agent_id not in agent_guard_dict:
                    continue

                unchecked_cache_guards = [g[:-1] for g in cached_guards[agent_id] if g[-1] < idx]     # FIXME: off by 1?
                for guard_expression, continuous_variable_updater, discrete_variable_dict, path in agent_guard_dict[agent_id] + unchecked_cache_guards:
                    assert isinstance(path, ModePath)
                    new_cont_var_dict = copy.deepcopy(cont_vars)
                    one_step_guard: GuardExpressionAst = copy.deepcopy(guard_expression)

                    self.apply_cont_var_updater(new_cont_var_dict, continuous_variable_updater)
                    guard_can_satisfied = one_step_guard.evaluate_guard_hybrid(
                        agent, discrete_variable_dict, new_cont_var_dict, track_map)
                    if not guard_can_satisfied:
                        continue
                    guard_satisfied, is_contained = one_step_guard.evaluate_guard_cont(
                        agent, new_cont_var_dict, track_map)
                    any_contained = any_contained or is_contained
                    # TODO: Can we also store the cont and disc var dict so we don't have to call sensor again?
                    if guard_satisfied:
                        reset_expr = ResetExpression((path.var, path.val_veri))
                        resets[reset_expr.var].append(
                            (reset_expr, discrete_variable_dict,
                             new_cont_var_dict, guard_expression.guard_idx, path)
                        )
                # Perform combination over all possible resets to generate all possible real resets
                combined_reset_list = list(itertools.product(*resets.values()))
                if len(combined_reset_list) == 1 and combined_reset_list[0] == ():
                    continue
                for i in range(len(combined_reset_list)):
                    # Compute reset_idx
                    reset_idx = []
                    for reset_info in combined_reset_list[i]:
                        reset_idx.append(reset_info[3])
                    # a list of reset expression
                    hits.append((agent_id, tuple(reset_idx), combined_reset_list[i]))
            if len(asserts) > 0:
                return (asserts, idx), None
            if hits != []:
                guard_hits.append((hits, state_dict, idx))
                guard_hit = True
            elif guard_hit:
                break
            if any_contained:
                break

        reset_dict = {}  # defaultdict(lambda: defaultdict(list))
        for hits, all_agent_state, hit_idx in guard_hits:
            for agent_id, reset_idx, reset_list in hits:
                # TODO: Need to change this function to handle the new reset expression and then I am done
                dest_list, reset_rect = self.apply_reset(node.agent[agent_id], reset_list, all_agent_state, track_map)
                # pp(("dests", dest_list, *[astunparser.unparse(reset[-1].val_veri) for reset in reset_list]))
                if agent_id not in reset_dict:
                    reset_dict[agent_id] = {}
                if not dest_list:
                    warnings.warn(
                        f"Guard hit for mode {node.mode[agent_id]} for agent {agent_id} without available next mode")
                    dest_list.append(None)
                if reset_idx not in reset_dict[agent_id]:
                    reset_dict[agent_id][reset_idx] = {}
                for dest in dest_list:
                    if dest not in reset_dict[agent_id][reset_idx]:
                        reset_dict[agent_id][reset_idx][dest] = []
                    reset_dict[agent_id][reset_idx][dest].append((reset_rect, hit_idx, reset_list[-1]))

        possible_transitions = []
        # Combine reset rects and construct transitions
        for agent in reset_dict:
            for reset_idx in reset_dict[agent]:
                for dest in reset_dict[agent][reset_idx]:
                    reset_data = tuple(map(list, zip(*reset_dict[agent][reset_idx][dest])))
                    paths = [r[-1] for r in reset_data[-1]]
                    transition = (agent, node.mode[agent],dest, *reset_data[:-1], paths)
                    src_mode = node.get_mode(agent, node.mode[agent])
                    src_track = node.get_track(agent, node.mode[agent])
                    dest_mode = node.get_mode(agent, dest)
                    dest_track = node.get_track(agent, dest)
                    if not track_map or dest_track == track_map.h(src_track, src_mode, dest_mode):
                        possible_transitions.append(transition)
        # Return result
        return None, possible_transitions

    def compute_full_reachtube(
        self,
        init_list: List[float],
        init_mode_list: List[str],
        static_list: List[str],
        uncertain_param_list: List[float],
        agent_list,
        transition_graph,
        time_horizon,
        time_step,
        lane_map,
        init_seg_length,
        reachability_method,
        run_num,
        past_runs,
        params = {},
    ):
        root = AnalysisTreeNode(
            trace={},
            init={},
            mode={},
            static = {},
            uncertain_param={},
            agent={},
            assert_hits={},
            child=[],
            start_time = 0,
            ndigits = 10,
            type = 'simtrace',
            id = 0
        )
        # root = AnalysisTreeNode()
        for i, agent in enumerate(agent_list):
            root.init[agent.id] = [init_list[i]]
            init_mode = [elem.name for elem in init_mode_list[i]]
            root.mode[agent.id] = init_mode
            init_static = [elem.name for elem in static_list[i]]
            root.static[agent.id] = init_static
            root.uncertain_param[agent.id] = uncertain_param_list[i]
            root.agent[agent.id] = agent
            root.type = 'reachtube'
        verification_queue = []
        verification_queue.append(root)
        num_calls = 0
        num_transitions = 0
        while verification_queue != []:
            node: AnalysisTreeNode = verification_queue.pop(0)
            combined_inits = {a: combine_all(inits) for a, inits in node.init.items()}
            print(node.mode)
            # pp(("start sim", node.start_time, {a: (*node.mode[a], *combined_inits[a]) for a in node.mode}))
            remain_time = round(time_horizon - node.start_time, 10)
            if remain_time <= 0:
                continue
            num_transitions += 1
            cached_tubes = {}
            # For reachtubes not already computed
            # TODO: can add parallalization for this loop
            for agent_id in node.agent:
                mode = node.mode[agent_id]
                inits = node.init[agent_id]
                combined = combine_all(inits)
                if self.config.incremental:
                    cached = self.trans_cache.check_hit(agent_id, mode, combined, node.init)
                    if cached != None:
                        self.trans_cache_hits = self.trans_cache_hits[0] + 1, self.trans_cache_hits[1]
                    else:
                        self.trans_cache_hits = self.trans_cache_hits[0], self.trans_cache_hits[1] + 1
                    # pp(("check hit", agent_id, mode, combined))
                    if cached != None:
                        cached_tubes[agent_id] = cached
                if agent_id not in node.trace:
                    # Compute the trace starting from initial condition
                    uncertain_param = node.uncertain_param[agent_id]
                    # trace = node.agent[agent_id].TC_simulate(mode, init, remain_time,lane_map)
                    # trace[:,0] += node.start_time
                    # node.trace[agent_id] = trace.tolist()
                    if reachability_method == "DRYVR":
                        # pp(('tube', agent_id, mode, inits))
                        cur_bloated_tube = self.calculate_full_bloated_tube(agent_id,
                                            mode,
                                            inits,
                                            remain_time,
                                            time_step, 
                                            node.agent[agent_id].TC_simulate,
                                            params,
                                            100,
                                            SIMTRACENUM,
                                            combine_seg_length=init_seg_length,
                                            lane_map = lane_map
                                            )
                    elif reachability_method == "NeuReach":
                        from verse.analysis.NeuReach.NeuReach_onestep_rect import calculate_bloated_tube_NeuReach
                        cur_bloated_tube = calculate_bloated_tube_NeuReach(
                            mode, 
                            inits[0], 
                            remain_time, 
                            time_step, 
                            node.agent[agent_id].TC_simulate, 
                            lane_map,
                            params, 
                        )
                    elif reachability_method == "MIXMONO_CONT":
                        cur_bloated_tube = calculate_bloated_tube_mixmono_cont(
                            mode, 
                            inits, 
                            uncertain_param, 
                            remain_time,
                            time_step, 
                            node.agent[agent_id],
                            lane_map
                        )
                    elif reachability_method == "MIXMONO_DISC":
                        cur_bloated_tube = calculate_bloated_tube_mixmono_disc(
                            mode, 
                            inits, 
                            uncertain_param,
                            remain_time,
                            time_step,
                            node.agent[agent_id],
                            lane_map
                        ) 
                    else:
                        raise ValueError(f"Reachability computation method {reachability_method} not available.")
                    num_calls += 1
                    trace = np.array(cur_bloated_tube)
                    trace[:, 0] += node.start_time
                    node.trace[agent_id] = trace.tolist()
            # pp(("cached tubes", cached_tubes.keys()))
            node_ids = list(set((s.run_num, s.node_id) for s in cached_tubes.values()))
            # assert len(node_ids) <= 1, f"{node_ids}"
            new_cache, paths_to_sim = {}, []
            if len(node_ids) == 1 and len(cached_tubes.keys()) == len(node.agent):
                old_run_num, old_node_id = node_ids[0]
                if old_run_num != run_num:
                    old_node = find(past_runs[old_run_num].nodes, lambda n: n.id == old_node_id)
                    assert old_node != None
                    new_cache, paths_to_sim = to_simulate(old_node.agent, node.agent, cached_tubes)
                    # pp(("to sim", new_cache.keys(), len(paths_to_sim)))

            # Get all possible transitions to next mode
            asserts, all_possible_transitions = self.postDisc(node, transition_graph.map, transition_graph.sensor, new_cache, paths_to_sim)
            # asserts, all_possible_transitions = transition_graph.get_transition_verify(new_cache, paths_to_sim, node)
            # pp(("transitions:", [(t[0], t[2]) for t in all_possible_transitions]))
            node.assert_hits = asserts
            if asserts != None:
                asserts, idx = asserts
                for agent in node.agent:
                    node.trace[agent] = node.trace[agent][:(idx + 1) * 2]
                continue

            transit_map = {k: list(l) for k, l in itertools.groupby(all_possible_transitions, key=lambda p:p[0])}
            transit_agents = transit_map.keys()
            # pp(("transit agents", transit_agents))
            if self.config.incremental and len(all_possible_transitions) > 0:
                transit_ind = max(l[-2][-1] for l in all_possible_transitions)
                for agent_id in node.agent:
                    transition = transit_map[agent_id] if agent_id in transit_agents else []
                    if agent_id in cached_tubes:
                        cached_tubes[agent_id].transitions.extend(convert_reach_trans(agent_id, transit_agents, node.init, transition, transit_ind))
                        pre_len = len(cached_tubes[agent_id].transitions)
                        cached_tubes[agent_id].transitions = dedup(cached_tubes[agent_id].transitions, lambda i: (i.mode, i.dest, i.inits))
                        # pp(("dedup!", pre_len, len(cached_tubes[agent_id].transitions)))
                    else:
                        self.trans_cache.add_tube(agent_id, combined_inits, node, transit_agents, transition, transit_ind, run_num)

            max_end_idx = 0
            for transition in all_possible_transitions:
                # Each transition will contain a list of rectangles and their corresponding indexes in the original list
                # if len(transition) != 6:
                #     pp(("weird trans", transition))
                transit_agent_idx, src_mode, dest_mode, next_init, idx, path = transition
                start_idx, end_idx = idx[0], idx[-1]

                truncated_trace = {}
                for agent_idx in node.agent:
                    truncated_trace[agent_idx] = node.trace[agent_idx][start_idx*2:]
                if end_idx > max_end_idx:
                    max_end_idx = end_idx

                if dest_mode is None:
                    continue

                next_node_mode = copy.deepcopy(node.mode)
                next_node_static = node.static
                next_node_uncertain_param = node.uncertain_param
                next_node_mode[transit_agent_idx] = dest_mode
                next_node_agent = node.agent
                next_node_start_time = list(truncated_trace.values())[0][0][0]
                next_node_init = {}
                next_node_trace = {}
                for agent_idx in next_node_agent:
                    if agent_idx == transit_agent_idx:
                        next_node_init[agent_idx] = next_init
                    else:
                        next_node_init[agent_idx] = [[truncated_trace[agent_idx][0][1:], truncated_trace[agent_idx][1][1:]]]
                        # pp(("infer init", agent_idx, next_node_init[agent_idx]))
                        next_node_trace[agent_idx] = truncated_trace[agent_idx]

                tmp = AnalysisTreeNode(
                    trace=next_node_trace,
                    init=next_node_init,
                    mode=next_node_mode,
                    static = next_node_static,
                    uncertain_param = next_node_uncertain_param,
                    agent=next_node_agent,
                    assert_hits = {},
                    child=[],
                    start_time=round(next_node_start_time, 10),
                    type='reachtube'
                )
                node.child.append(tmp)
                verification_queue.append(tmp)

            """Truncate trace of current node based on max_end_idx"""
            """Only truncate when there's transitions"""
            if all_possible_transitions:
                for agent_idx in node.agent:
                    node.trace[agent_idx] = node.trace[agent_idx][:(
                        max_end_idx+1)*2]

        self.reachtube_tree = AnalysisTree(root)
        # print(f">>>>>>>> Number of calls to reachability engine: {num_calls}")
        # print(f">>>>>>>> Number of transitions happening: {num_transitions}")
        self.num_transitions = num_transitions

        return self.reachtube_tree


