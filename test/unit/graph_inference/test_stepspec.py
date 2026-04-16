"""
Integration tests for StepSpec — actual flow execution.
"""


class TestStepSpecSimpleExecution:
    def test_flow_completes(self, stepspec_simple_run):
        assert stepspec_simple_run.successful
        assert stepspec_simple_run.finished

    def test_graph_info_endpoints(self, stepspec_simple_run):
        graph_info = stepspec_simple_run["_parameters"].task["_graph_info"].data
        assert graph_info["start_step"] == "stepspecsimpleflow"
        assert graph_info["end_step"] == "stepspecsimpleflow"

    def test_end_task_data(self, stepspec_simple_run):
        end_task = stepspec_simple_run.end_task
        assert end_task is not None
        assert end_task["output"].data == "HELLO"

    def test_single_step_present(self, stepspec_simple_run):
        step_names = {s.id for s in stepspec_simple_run}
        assert step_names == {"stepspecsimpleflow"}


class TestStepSpecInitExecution:
    def test_flow_completes(self, stepspec_init_run):
        assert stepspec_init_run.successful
        assert stepspec_init_run.finished

    def test_init_and_call_ran(self, stepspec_init_run):
        end_task = stepspec_init_run.end_task
        assert end_task is not None
        # factor=2 (default), so computed = 2*10 = 20
        # value=10 (default), so result = 10 * 20 = 200
        assert end_task["result"].data == 200

    def test_init_state_persisted(self, stepspec_init_run):
        end_task = stepspec_init_run.end_task
        assert end_task["computed"].data == 20
