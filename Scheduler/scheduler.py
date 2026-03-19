# This code defines the Scheduler class, which simulates an inference serving system on top of TOGSim.
# Scheduler receives inference requests (with arrival times), batches them by model, and dispatches
# them to PyTorchSimRunner which compiles each model into a sequence of kernels.
# Kernels are then executed one by one through TOGSim in FIFO or Round-Robin order,
# enabling performance evaluation metrics such as response time, turnaround time, and TBT (Time Between Tokens).

from typing import List
import os
import sys
import numpy as np
import torch
from pathlib import Path
import importlib.util
from PyTorchSimFrontend.extension_codecache import hash_prefix
from Simulator.simulator import TOGSimulator
from PyTorchSimFrontend import extension_config

# Configure logger for Scheduler module
logger = extension_config.setup_logger()


def import_module_from_path(module_name, path):
    module_path = Path(path)  # Convert to Path object for safety
    if not module_path.exists() or not module_path.is_file():
        raise FileNotFoundError(f"No such file: '{module_path}'")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Could not load module from path: '{module_path}'")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module

def poisson_request_generator(lambda_requests, max_msec_time=None):
    current_time = 0.0 # msec

    yield 0
    while max_msec_time is None or current_time < max_msec_time:
        inter_arrival_time = np.random.exponential(scale=1000 / lambda_requests)
        current_time += inter_arrival_time

        if max_msec_time is not None and current_time > max_msec_time:
            break

        yield current_time


# Request class represents each request which is sent to device and becomes workload
# Request objects will be batched(by same model) and sent to ExecutionEngine
class Request:
    """ Each request has model name, it's own id, and requested time. """
    request_id = 0
    QUEUED     = 1
    RUNNING    = 2
    INCREMENT  = 3
    FINISHED   = 4
    # Initialize all attributes
    def __init__(self, model:str, batchable_input_tensor : List[torch.Tensor],
                 shared_input_tensor: List[torch.tensor], request_queue_idx=0) -> None:
        self.model = model # Of which model the request belongs to (BERT, convolution, RESNET50 etc)
        self.batchable_input_tensor = batchable_input_tensor # input_tensor a request holds that can be later batched with other request's batchable_input_tensor (ex: KV cache, input token)
        self.shared_input_tensor = shared_input_tensor # input_tensor a request holds that can be shared with other requests (ex: model weights)
        self.arrival_time = None # Time at which the request first arrived
        self.start_time = [] # List of times at which the execution of request has started. It is a list because a reqeust can pause mid-execution and be rescheduled later. 
        self.finish_time = [] # List of times at which the execution of request has finished
        self.state = self.QUEUED # State of the request → QUEUED, RUNNING, INCREMENT, FINISHED
        self.id = self.allocate_id() # Unique ID of the request
        self.request_queue_idx = request_queue_idx # Of which request queue (= parition) the request belongs to

    # Make unique id of the request
    def allocate_id(self):
        allocated_id = Request.request_id
        Request.request_id += 1
        return allocated_id

    # Start the execution of the request and append the time to `start_time` list
    def set_start(self, start_time):
        self.state = self.RUNNING
        self.start_time.append(start_time)

    # List of times at which the execution of request has finished
    def set_finished(self, finish_time):
        self.state = self.FINISHED
        self.finish_time.append(finish_time)

    # Get turnaround, response, tbt time of the request
    def get_latency(self):
        # Todo. Provide Toke-By-Token
        if self.state == self.FINISHED:
            turnaround_time = self.finish_time[-1] - self.arrival_time
        else:
            turnaround_time = None

        if self.start_time:
            response_time = self.start_time[0] - self.arrival_time
        else:
            response_time = None

        if self.start_time and self.finish_time:
            tbt_time = [i-j for i,j in zip(self.finish_time, self.start_time)]
        else:
            tbt_time = []

        return turnaround_time, response_time, tbt_time


    def free_memory(self):
        """ Free memory resources that are allocated for handle this request """
        return

    # Just for logging
    def __str__(self) -> str:
        return f"Request{self.id} Model: '{self.model}', Arrival: {self.arrival_time}, Start: {self.start_time}, End: {self.finish_time}, State: {self.state}, Partion: {self.request_queue_idx}"

# Class for returned(finished) requests
# Identifies which state the returned request has 
class RequestReturn:
    INCREMENT = 0
    FINISHED = 1
    def __init__(self, state) -> None:
        self.state = state

    def is_finished(self):
        return self.state == self.FINISHED

    def is_increment(self):
        return self.state == self.INCREMENT


# Represent each model to which each request belongs
# Each model holding several requests will be run in one partition of the device
# The name is model but we can interpret this as a layer(such as attention, MLP, CNN etc)
class SchedulerDNNModel:
    # Dictionary holding pairs of {model name: compiled model} 
    MODEL_MAP = {}

    def __init__(self, batched_req : List[Request], partition_idx) -> None:
        self.model_name = batched_req[0].model # String of model name of batched request
        self.batched_req = batched_req # Batched requests which SchedulerDNNModel holds
        self.args = None 

        # compiled_model object for corresponding model name
        # compiled_modelis torch._dynamo.OptimizedModule or compiled callable which went through PyTorch compiler optimization
        # For requests with different tensor shapes, the model is recompiled since dynamic=False
        # In test codes, model corresponds to each layer. When experimenting, divide the total model into layers, and generate=compile modelper layer and run them separately.
        self.model = self.find_model(self.model_name)
        # In which partition the model will be run
        self.partition_idx = partition_idx

    # From MODEL_MAP return compiled_model object matching the model name
    def find_model(self, model_name : str):
        if model_name in SchedulerDNNModel.MODEL_MAP:
            return SchedulerDNNModel.MODEL_MAP[model_name]
        else:
            raise KeyError(f'[Scheduler] Requested model "{model_name}" is not registered...')

    # Each SchedulerDNNModel object has several requests and requests have batchable input tensors
    # This function batches up batchable input tensors across several requests. 
    def get_batchable_input(self):
        batched_input_tensor = []
        for i in range(len(self.batched_req[0].batchable_input_tensor)):
            tensor_list = [req.batchable_input_tensor[i] for req in self.batched_req]
            batched_input_tensor.append(torch.concat(tensor_list, dim=0))
        return batched_input_tensor

    # Get shared_input_tensor which all requests inside SchedulerDNNModel object shares
    def get_shared_input(self):
        return self.batched_req[0].shared_input_tensor

    # Get all input tensors including batched input tensors and shared tensors
    def get_input(self):
        return self.get_batchable_input() + self.get_shared_input()

    def __str__(self):
        return f"DNN Model: {self.model_name}, Partion idx: {self.partition_idx} Req: {self.batched_req[0]}"

    @staticmethod
    def register_model(model_name : str, compiled_model):
        SchedulerDNNModel.MODEL_MAP[model_name] = compiled_model


# Engine part of scheduler, which is in charge of scheduling and sending requests to simulator. Define methods to execute kernels(==models). 
class PyTorchSimRunner:
    PARTITION_BUSY = 0
    PARTITION_IDLE = 1
    SELECT_NOTHING = 2
    NPU_MODULE = None

    # Initializes attributes and make launch_model_dicts, nested_launch_model_dicts, partition_state per partition
    def __init__(self, tog_simulator : TOGSimulator, num_partion=1) -> None:
        self.module = self.setup_device() # Configured device which Pytorch kernel will be executed on 
        self.num_partion = num_partion # Number of partitions in a device

        # List of dictionaries of {SchedulerDNNModel(==compiled_model): generator(which returns (kernel_fn, inputs) where kernel_fn when called produces the ONNX graph + timing attributes for TOGSim)}
        # Each dictionary represents models that should be run in one partition
        # {SchedulerDNNModel: generator which returns each layer's kernels and inputs}
        # Generator will return each layer’s kernels and inputs when called, and kernel is an object compiled by Pytorch Inductor representing each computation
        self.launch_model_dicts = [] 
        self.nested_launch_model_dicts = [] # Same with launch_model_dicts  ut a model nests several models inside 
        self.partition_state = [] List of states of each partition
        for i in range(self.num_partion):
            self.launch_model_dicts.append({})
            self.nested_launch_model_dicts.append({})
            self.partition_state.append(self.PARTITION_IDLE)

        self.finish_req_dict = {}
        self.tog_simulator = tog_simulator # TOGSim object which is the very core simulator part running and producing results

        # Dry run for compile and create generator
        os.environ["TOGSIM_EAGER_MODE"] = "1"

    @classmethod
    # Register device backend information of custom NPU into Pytorch Inductor's compiler == Enable torch.device("npu")
    def setup_device(cls):
        if cls.NPU_MODULE is not None:
            return cls.NPU_MODULE

        try:
            from torch._inductor.codegen.common import register_backend_for_device
            from PyTorchSimFrontend.mlir.mlir_codegen_backend import ExtensionWrapperCodegen
            from PyTorchSimFrontend.mlir.mlir_scheduling import MLIRScheduling
        except ImportError as e:
            logger.error(f"Failed to import torch_openreg: {e}")
            logger.error("Please ensure PyTorchSimDevice2 is installed: pip install -e PyTorchSimDevice2")
            raise

        register_backend_for_device(
            "npu",
            lambda scheduling: MLIRScheduling(scheduling),
            ExtensionWrapperCodegen
        )

        cls.NPU_MODULE = torch.npu
        return cls.NPU_MODULE

    # Submit batched requests and save models to be run into launch_model_dicts, which holds {SchedulerDNNModel == compiled_model : generator(kernels and input tensors)}
    def submit(self, batched_req, partition_idx) -> List[RequestReturn]:
        # FIXME. Construct SchedulerDNNModel
        batched_req_model = self.get_compiled_model(batched_req, partition_idx)
        self.prepare_model(batched_req_model)

    # Compile batched requests into SchedulerDNNModel object 
    def get_compiled_model(self, batched_req: List[Request], request_queue_idx):
        compiled_model = SchedulerDNNModel(batched_req, request_queue_idx)
        return compiled_model

    # Return whether a partition is idle(not working)
    def is_partition_idle(self, partition_idx):
        return len(self.launch_model_dicts[partition_idx]) == 0

    # Return whether any parition is idle(not working)
    def is_any_idle(self, skip_list):
        return any([self.is_partition_idle(i) and not skip_list[i] for i in range(self.num_partion)])

    # Return whether all paritions are idle(not working)
    def is_all_idle(self):
        return all([self.is_partition_idle(i) for i in range(self.num_partion)])

    # Prepare SchedulerDNNModel object by getting generators for each compiled_model and inserting into launch_model_dicts
    # Prepare result_path of each SchedulerDNNModel and input tensors
    # Get generator(kernels + input tensor) for each SchedulerDNNModel object and update to corresponding partition's launch_model_dicts
    # This updates launch_model_dicts[partition_idx], which is used in select_kernel of FIFORunner or RoundRobinRunner later, which will execute each parition's model's kernels
    def prepare_model(self, req_model: SchedulerDNNModel):
        result_path = os.path.join(extension_config.CONFIG_TORCHSIM_LOG_PATH, "togsim_result", req_model.model_name)
        os.makedirs(result_path, exist_ok=True)
        index = str(len(os.listdir(result_path)))

        # Prepare input tensor
        input_tensor_list = req_model.get_input()
        input_tensor_list = [input_tensor.to(device=self.module.custom_device()) for input_tensor in input_tensor_list]

        # This model-call will return generator
        ret = req_model.model(*input_tensor_list)
        self.launch_model_dicts[req_model.partition_idx][req_model] = ret

    # Mark every request finished in a SchedulerDNNModel object
    def finish_model(self, model : SchedulerDNNModel, output : torch.Tensor):
        for req in model.batched_req:
            # TODO. finish time
            self.finish_req_dict[req] = RequestReturn(RequestReturn.FINISHED)

    # Prepare onnx path and attribute path for each kernel and input so that kernel can be laucnhed
    # Kernel is callable object, which give pathes by calling kernel(*inputs) 
    def prepare_launch_kernel(self, kernel, inputs):
        result_path, runtime_path, _ = kernel(*inputs)
        onnx_path = os.path.join(result_path, "tile_graph.onnx")

        attribute_path = os.path.join(runtime_path, "attribute")
        attribute_path = self.tog_simulator.create_attribute_file(attribute_path, inputs)
        return onnx_path, attribute_path


    # Launch kernel in corresponding partition using TOGSim
    # Kernels and inputs are converted into onnx_path(TOG) and attribute_path by prepare_launch_kernel() and passed to TOGSim
    # If string, just get the path string to onnx_path and attribute_path(indicating file) and if not, call prepare_launch_kernel(self,kernel, inputs), eventually getting pathes anyway
    # Then, send launch command with onnx_path, attribute_path, current_cycle, and partition_idx to TOGSim to execute simulation
    def launch_kernel(self, current_cycle, partion_idx=0):
        # Check partition is busy
        if self.partition_state[partion_idx] != self.PARTITION_IDLE:
            return self.partition_state[partion_idx]
        result = self.select_kernel(partion_idx)
        if result == self.SELECT_NOTHING:
            return self.SELECT_NOTHING
        kernel, inputs = result
        if not isinstance(kernel, str):
            onnx_path, attribute_path = self.prepare_launch_kernel(kernel, inputs)
        else:
            onnx_path, attribute_path = kernel, inputs
        self.partition_state[partion_idx] = self.PARTITION_BUSY
        return self.tog_simulator.launch(onnx_path, attribute_path, current_cycle, partion_idx)



# FIFO version of PyTorchSimRunner. select_kernel() logic is in FIFO
class FIFORunner(PyTorchSimRunner):
    def __init__(self, tog_simulator: TOGSimulator, num_partion=1) -> None:
        super().__init__(tog_simulator, num_partion)

    # Select kernel and input tensors from generator inside nested_launch_model_dicts or launch_model_dicts while iterating in FIFO manner
    def select_kernel(self, partition_idx):
        while len(self.nested_launch_model_dicts[partition_idx]) or len(self.launch_model_dicts[partition_idx]):
            if len(self.nested_launch_model_dicts[partition_idx]):
                target_dict = self.nested_launch_model_dicts
            else:
                target_dict = self.launch_model_dicts

            # Select FIFO manner
            req, target_model = next(iter(target_dict[partition_idx].items()))
            try:
                kernel, inputs = next(target_model)

                # For extern call
                if isinstance(kernel, str):
                    return kernel, inputs

                # For convolution...
                if not hasattr(kernel, "future"):
                    nested_gen = kernel(*inputs)
                    self.nested_launch_model_dicts[partition_idx] = {req : nested_gen}
                    kernel, inputs = \
                        next(self.nested_launch_model_dicts[partition_idx][req])
                return kernel, inputs
            except StopIteration as e:
                # Retry
                if target_dict == self.launch_model_dicts:
                    self.finish_model(req, e.value)
                del target_dict[partition_idx][req]
        # No proper kernel now
        return self.SELECT_NOTHING

# RR version of PyTorchSimRunner. select_kernel()logic is in RR.
class RoundRobinRunner(PyTorchSimRunner):
    def __init__(self, tog_simulator: TOGSimulator, num_partion=1) -> None:
        super().__init__(tog_simulator, num_partion)
        self.next_pointer = None

    # Iterate nested_launch_model_dicts or launch_model_dicts but alternating model
    # In other words, request from different model is executed alternately every time
    def select_kernel(self, partition_idx):
        while len(self.nested_launch_model_dicts[partition_idx]) or len(self.launch_model_dicts[partition_idx]):
            if len(self.nested_launch_model_dicts[partition_idx]):
                target_dict = self.nested_launch_model_dicts
            else:
                target_dict = self.launch_model_dicts

            req_list = list(target_dict[partition_idx].keys())
            # Select RR manner
            if self.next_pointer is None or self.next_pointer not in req_list:
                req = req_list[0]
                pos = 0
            else:
                req = self.next_pointer
                pos = req_list.index(req)

            # Set Next pointer
            if pos + 1 < len(req_list):
                self.next_pointer = req_list[pos+1]
            else:
                self.next_pointer = req_list[0]

            target_model = self.launch_model_dicts[partition_idx][req]
            try:
                kernel, inputs = next(target_model)

                # For convolution...
                if not hasattr(kernel, "future"):
                    nested_gen = kernel(*inputs)
                    self.nested_launch_model_dicts[partition_idx] = {req : nested_gen}
                    kernel, inputs = \
                        next(self.nested_launch_model_dicts[partition_idx][req])
                return kernel, inputs
            except StopIteration as e:
                # Retry
                if target_dict == self.launch_model_dicts:
                    self.finish_model(req, e.value)
                del self.launch_model_dicts[partition_idx][req]
        # No proper kernel now
        return self.SELECT_NOTHING


# The main frontend component of TOGSim to run simulation
# Scheduler receive requests and batch them up according to same models and generate TOGSim object and the TOGSim object executes kernels one by one
class Scheduler:

    FIFO_ENGINE = 0
    RR_ENGINE = 1
    def __init__(self, num_request_queue=1, max_batch=1, engine_select=FIFO_ENGINE, togsim_config=extension_config.CONFIG_TOGSIM_CONFIG) -> None:
        self.current_cycle = 0 # Current cycle
        self.max_batch = max_batch # Maximum number of requests that can be inside one batch
        self.num_request_queue = num_request_queue # Number of request queue = partition
        self.request_queue : List[List[Request]] = [] # [Partition 1’s list of Request, Partition 2’s list of Request…..]
        for i in range(self.num_request_queue):
            self.request_queue.append([])
        self.finish_queue : List[Request] = [] # [Partition 1’s list of finished Request, Partition 2’s list of finished Request…..]


        # TOGSim object which runs simulation of kernels
        self.tog_simulator = TOGSimulator(togsim_config)
        if self.tog_simulator.config_yaml['pytorchsim_timing_mode'] == 0:
            # Scheduler requires timing mode to be enabled (pytorchsim_timing_mode != 0).
            logger.error(f"pytorchsim_timing_mode is set to 0 in config file '{togsim_config}'. ")
            logger.error(f"Scheduler requires timing mode to be enabled (pytorchsim_timing_mode != 0).")
            exit(0)

        os.environ['TOGSIM_CONFIG'] = togsim_config
        self.tog_simulator.interactive_simulation()

        # FIFORunner or RoundRobinRunner that will select kernels and launch to tog_simulator 
        if engine_select == Scheduler.FIFO_ENGINE:
            self.execution_engine = FIFORunner(self.tog_simulator, self.num_request_queue)
        elif engine_select == Scheduler.RR_ENGINE:
            self.execution_engine = RoundRobinRunner(self.tog_simulator, self.num_request_queue)
        else:
            logger.error(f"Not supported engine type {engine_select}")
            exit(1)

    # Add request to each partition’s request queue and update request’s arrival time
    def add_request(self, request: Request, request_time=-1):
        """register model at timestamp time
            request_time : msec
        """
        request_time = self.current_time() if request_time == -1 else request_time
        request.arrival_time = request_time
        self.request_queue[request.request_queue_idx].append(request)

    # Return whether reqeust_queue of a partition is empty
    def request_empty(self, request_queue_idx):
        return len(self.request_queue[request_queue_idx])==0

    # From request queue of one partition, batch up requests to candidate_req and return
    def select(self, request_queue_idx=0) -> List[Request]:
        """
        Select 1 request from request_queue in FCFS manner.
        If there is no proper request, return None
        """
        candidate_req = []
        if not self.request_queue[request_queue_idx]:
            return candidate_req
        for req in self.request_queue[request_queue_idx]:

            if self.msec_to_cycle(req.arrival_time) <= self.current_cycle and req.state == Request.QUEUED:
                candidate_req.append(req)

                # Stop batching
                if self.max_batch <= len(candidate_req):
                    break
        return candidate_req

    # Return the first queueing reqeust and request arrival time inside one request_queue of one partition
    def next_request_time(self, request_queue_idx=0):
        for req in self.request_queue[request_queue_idx]:
            if req.state == Request.QUEUED:
                return req, req.arrival_time
        return None, -1

    # From all request_queue of all partitions, get nearest queueing request(in terms of arrival time) and it's arrival time
    def nearest_next_reqeust_time(self):
        nearest_req = None
        nearest_arrival_time = -1
        for i in range(self.num_request_queue):
            req, arrival_time = self.next_request_time(i)
            if nearest_arrival_time == -1 and arrival_time != -1:
                nearest_req = req
                nearest_arrival_time = arrival_time
            elif arrival_time != -1 and nearest_arrival_time > arrival_time:
                nearest_req = req
                nearest_arrival_time = arrival_time
        return nearest_req, nearest_arrival_time

    # Finish request and free resources and add the request to finish_queue list
    def finish_request(self, req : Request):
        req.set_finished(self.current_time())

        # Free resources
        req.free_memory()

        # Move to finish queue
        self.finish_queue.append(req)
        self.request_queue[req.request_queue_idx].remove(req)
        turnaround_time, response_time, tbt_time = req.get_latency()
        logger.info(
            f"[Request-{req.id} finished] partition: {req.request_queue_idx} arrival_time: "
            f"{req.arrival_time} start_time: {req.start_time[0]} turnaround latency: {turnaround_time}, "
            f"response time: {response_time} tbt_time: {tbt_time}"
        )

    # For one partition, choose a batch of batched requests(request_list) and call execution_engine.start(request_list, requeust_queue_idx)
    # This will then call prepare_model(), which update each partition’s launch_model_dicts, which has generators of kernels and input tensors of each model
    def per_schedule(self, request_queue_idx):
        # Wait partition is idle
        if not self.execution_engine.is_partition_idle(request_queue_idx):
            return False

        request_list = self.select(request_queue_idx)
        if not request_list:
            return False

        logger.info(f"[Request issue] partition: {request_queue_idx} batch size: {len(request_list)}")
        for req in request_list:
            req.set_start(self.current_time())
            logger.info(
                f"[Request-{req.id} issue] partition: {req.request_queue_idx} "
                f"arrival_time: {req.arrival_time} start_time: {req.start_time[0]}"
            )
        # Submit batched request
        self.execution_engine.submit(request_list, request_queue_idx)

        return True

    # 
    def check_finish_request(self):
        # Check finished request
        while self.execution_engine.finish_req_dict:
            req, req_ret = next(iter(self.execution_engine.finish_req_dict.items()))
            self.finish_request(req)
            del self.execution_engine.finish_req_dict[req]

    # For every partition, run per_schedule(), which will choose a batch of requests and call prepare_model
    # If no request is left and all partitions are idle → return
    # Else if all partitions are idle but have requests to do → run simulation until next request == jump cycles toward next request
    # Else if there exists request running in partition → run(next_time) == continue simulation for running request until next_time
    def schedule(self):
        # Try schedule all request queue
        result = []
        for i in range(self.num_request_queue):
            result.append(self.per_schedule(i))

        # Try move to next nearest request time
        next_req, next_time = self.nearest_next_reqeust_time()
        if next_req is None and self.execution_engine.is_all_idle():
            # No request remained...
            return

        # Need to forward the time until next_arrival_time
        if self.execution_engine.is_all_idle():
            reason = self.tog_simulator.until(self.msec_to_cycle(next_time))
            self.current_cycle = self.tog_simulator.cycle()
        else:
            self.run(next_time)
        return

    # Run each kernel in each partition using tog_simulator and update cycles
    # until_time == - 1 → Just run simulation until all partition is idle or request queue is empty 
    # until_time ≠ -1 → Run simulation until until_time. If all parition is idle(all requests done), should stop
    def run(self, until_time):
        req_empty_info = [self.request_empty(i) for i in range(self.execution_engine.num_partion)]
        def execute_cycle():
            launch_ret_info = []
            for i in range(self.execution_engine.num_partion):
                if self.execution_engine.partition_state[i] == PyTorchSimRunner.PARTITION_IDLE:
                    ret = self.execution_engine.launch_kernel(self.current_cycle, i)
                    launch_ret_info.append(ret)

            self.check_finish_request()
            # Check if the stop condition is met
            if self.execution_engine.is_any_idle(req_empty_info) or self.execution_engine.is_all_idle(): # Ignore empty request queue
                return []

            # Schedule jobs and update the current time
            result_list = self.tog_simulator.until(self.msec_to_cycle(until_time))
            self.current_cycle = self.tog_simulator.cycle()

            for core_idx in result_list:
                # Kernel is finished. So set idle state
                self.execution_engine.partition_state[core_idx] = PyTorchSimRunner.PARTITION_IDLE

            return result_list

        if self.current_cycle >= self.msec_to_cycle(until_time):
            until_time = -1

        if until_time == -1:
            while not self.execution_engine.is_any_idle(req_empty_info):
                result = execute_cycle()
                req_empty_info = [self.request_empty(i) for i in range(self.execution_engine.num_partion)]
                # if result is not -1, schedule new request
                if len(result)==0:
                    break

        else:
            while self.current_cycle <= self.msec_to_cycle(until_time) and not self.execution_engine.is_all_idle():
                result = execute_cycle()
                # if result is not -1, schedule new request
                if len(result)==0:
                    break
        return

    # If all request queues are empty and partitions are idle, then stop backend_simulator
    def is_request_queue_empty(self):
        result = True
        for i in range(self.num_request_queue):
            result = result and (not len(self.request_queue[i]))
        return result

    # If all request queues are empty and partitions are idle, then stop backend_simulator
    def is_finished(self):
        if self.is_request_queue_empty() and self.execution_engine.is_all_idle():
            self.tog_simulator.wait()
            return True
        return False

    def current_time(self):
        return self.cycle_to_msec(self.current_cycle)

    def cycle_to_msec(self, cycle):
        freq = self.tog_simulator.get_core_freq()
        return cycle / (freq  / 1000)

    def msec_to_cycle(self, msec):
        # We treat -1 as special time
        if (msec == -1):
            return msec

        freq = self.tog_simulator.get_core_freq()
        return int(msec * (freq / 1000))
