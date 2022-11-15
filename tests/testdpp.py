# Introducing unittests for DryVR++ development process
# Read more from https://docs.python.org/3/library/unittest.html

# A scenario is created for testing
from enum import Enum, auto
import unittest

from verse.scenario.scenario import Scenario
from verse.map.example_map.simple_map import SimpleMap2


class TestSimulatorMethods(unittest.TestCase):
    def setUp(self):
        pass
        # self.scenario = Scenario()
        # self.car = CarAgent('ego', file_name='example_controller1.py')
        # self.car2 = CarAgent('other', file_name='example_controller1.py')
        # self.scenario.add_agent(self.car)
        # self.scenario.add_agent(self.car2)
        # self.scenario.add_map(SimpleMap2())
        # self.scenario.set_sensor(FakeSensor1())
        # self.scenario.set_init(
        #     [[0, 3, 0, 0.5]],
        #     [(VehicleMode.Normal, TrackMode.Lane0)]
        # )
        # self.traces = scenario.simulate(
        #     10
        # )
        #
        # self.queue = [traces]

    def test_postCont_1(self):
        from verse.analysis.verifier import Verifier
        from origin_agent import vanderpol_agent
        from verse.analysis.analysis_tree import AnalysisTreeNode
        from verse.scenario import ScenarioConfig

        node = AnalysisTreeNode(
            init={'test':[[[1.25, 2.25], [1.25, 2.25]]]},
            mode={'test':'Normal'},
            agent={'test':vanderpol_agent('test')},
            start_time=0,
            type='reachtube',
        )

        remain_time = 7
        time_step = 0.05
        lane_map = None
        
        tmp_verifier = Verifier(ScenarioConfig())
        res = tmp_verifier.postCont(
            node,
            remain_time, 
            time_step, 
            lane_map,
            combine_seg_length = 1
        )
        assert len(res.trace)==1
        assert len(res.trace['test'])==280
        pass

    def test_postDisc_1(self):
        from verse.analysis.verifier import Verifier
        from origin_agent import thermo_agent
        from verse.analysis.analysis_tree import AnalysisTreeNode
        from verse.scenario import ScenarioConfig
        from thermo_controller import thermo_controller_codestring
        from verse.map.lane_map import LaneMap
        from verse.sensor.base_sensor import BaseSensor

        test_agent = thermo_agent('test',code = thermo_controller_codestring)
        node = AnalysisTreeNode(
            trace = {'test':[[0,60,10,1],[0.1,61,10,1]]},
            mode={'test':['ON']},
            static={'test':[]},
            agent={'test':test_agent},
        )

        tmp_verifier = Verifier(ScenarioConfig())
        asserts, all_possible_transitions = tmp_verifier.postDisc(
            node,
            None,
            BaseSensor(),
        )

        assert asserts is None 

        transition = all_possible_transitions[0]
        agent_id = transition[0]
        transition_src = transition[1]
        transition_dest = transition[2]
        reset_rect = transition[3][0]
        reset_idx = transition[4][0]

        assert agent_id=='test'
        assert transition_src == ['ON']
        assert transition_dest == ('OFF',)
        assert reset_rect == [[60,10,0],[61,10,0]]
        assert reset_idx == 0

if __name__ == "__main__":
    unittest.main()
