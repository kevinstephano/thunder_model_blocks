import argparse
from collections import OrderedDict
from functools import partial
from itertools import product
from nvfuser import FusionCache
import subprocess
import sys
import thunder
from thunder.dynamo import thunderfx
from time import sleep
import timeit
import torch
from torch.profiler import profile, ProfilerActivity
import traceback

from thunder_model_blocks.utils.gpu_utils import lock_max_gpu_clocks, reset_gpu_clocks
from thunder_model_blocks.utils.lora import patch_linear_module
from thunder_model_blocks.utils.packed_sequences import dummy_packed_seqs
#from nemo.collections.llm.peft.lora import patch_linear_module

def install_pandas():
    """
    Check if pandas is installed, and if not, install it using pip.
    Returns True if pandas is successfully installed or already present,
    False if installation fails.
    """
    try:
        import pandas as pd
        return pd
    except ImportError:
        print("pandas is not installed. Installing now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
            print("pandas has been successfully installed!")
            import pandas as pd
            return pd
        except subprocess.CalledProcessError:
            print("Failed to install pandas. Please try installing manually using:")
            print("pip install pandas")
            return None

def install_transformers():
    """
    Check if transformers is installed, and if not, install it using pip.
    Returns True if pandas is successfully installed or already present,
    False if installation fails.
    """
    try:
        import transformers as xfs
        return xfs
    except ImportError:
        print("HuggingFace Transformers is not installed. Installing now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
            print("transformerss has been successfully installed!")
            import transformers as xfs
            return xfs
        except subprocess.CalledProcessError:
            print("Failed to install transformers. Please try installing manually using:")
            print("pip install transformers")
            return None

def run(sys_argv, model_name, config, module, input_fn, module_has_loss=False, grad_fn=None, inference=False) : 
    pd = install_pandas()
    assert pd is not None, "Pandas is not installed!"
    xfs = install_transformers()
    assert xfs is not None, "Transformers is not installed!" 
    pd.options.display.max_colwidth=200
    pd.options.display.float_format = '{:.3f}'.format

    parser = argparse.ArgumentParser(description='Thunder Based Model Examples')
    parser.add_argument('--nsys', default=False, action="store_true", help='Disables torch.profiler for nsys.')
    parser.add_argument('--pivot', default=False, action="store_true", help='Create a pivot table of executors as columns that is more suitable for graphing.')
    parser.add_argument('--csv', default=False, action="store_true", help='Print csv output.')
    parser.add_argument('--warmup', default=10, type=int, help='Warmup iterations.')
    parser.add_argument('--dtype', default='bfloat16', type=str, help="Set model and activation data types.")
    parser.add_argument('--batch_sizes', nargs='*', default=None, type=int, help="List of batch sizes. The default of None says to use the model default.")
    parser.add_argument('--seq_lens', nargs='*', default=None, type=int, help="List of sequence lengths. The default of None says to use the model default.")
    parser.add_argument('--wallclock_iters', default=10, type=int, help='Wallclock Timing iterations.')
    parser.add_argument('--cupti_iters', default=1, type=int, help='CUPTI Timing iterations.')
    parser.add_argument('--execs', nargs='+', type=str, help='List of executor names to time.', required=False,
        default=["Torch-Eager", "torch.compile", "Thunder-torch.compile", "Thunder-nvFuser"],
        choices=["Torch-Eager", "torch.compile", "Thunder-Torch", "Thunder-torch.compile", "Thunder-default", "Thunder-nvFuser", "Thunder-nvFuser-more-ops"])
    parser.add_argument('--thunder_trace', default=False, action="store_true", help='Prints a Thunder trace.')
    parser.add_argument('--thunder_trace_for_error', default=False, action="store_true", help='Prints a Thunder trace prior to execution to capture a trace before it hits a runtime error.')
    parser.add_argument('--dynamo_explain', default=False, action="store_true", help='Prints an explanation of why dynamo generates graph breaks.')
    parser.add_argument('--nvfuser_repro', default=False, action="store_true", help='Prints an nvFuser reproduction script.')
    parser.add_argument('--attn', default='sdpa', type=str, help='Selects the type of Fused Attention.', choices=['sdpa', 'flash_attention', 'flash_attention_2', 'eager'])
    parser.add_argument('--lora', default=False, action="store_true", help='Enables Lora based PEFT benchmarking.')
    parser.add_argument('--two_layers', default=False, action="store_true", help='Change model size to only 2 hidden layers.')
    parser.add_argument('--packed_seqs', default=False, action="store_true", help='Enable packed sequences')
    parser.add_argument('--disable_locked_clocks', default=False, action="store_true", help='Disable Locked Clocks for debug.')
    args,extra_args = parser.parse_known_args(args=sys_argv[1:])

    assert len(extra_args) == 0, "Unknown args: {}".format(extra_args)

    def eager_wrapper(model):
        return model

    config._attn_implementation = args.attn
    config.lora = args.lora
    if args.two_layers:
        config.num_hidden_layers = 2

    if "/" in model_name:
        model_name = model_name.split("/")[1]

    executors = OrderedDict()
    for exec in args.execs:
        if exec == "Torch-Eager":
            executors["Torch-Eager"] = eager_wrapper
        elif exec == "Thunder-Torch":
            executors["Thunder-Torch"] = partial(thunderfx, executors=["torch"])
        elif exec == "torch.compile":
            executors["torch.compile"] = partial(torch.compile)
        elif exec == "Thunder-default":
            executors["Thunder-default"] = partial(thunderfx)
        elif exec == "Thunder-nvFuser":
            executors["Thunder-nvFuser"] = partial(thunderfx, executors=["apex", "cudnn","sdpa","nvfuser"])
        elif exec == "Thunder-nvFuser-more-ops":
            executors["Thunder-nvFuser-more-ops"] = partial(thunderfx, executors=["cudnn","sdpa","nvfuser"], nv_enable_linear=True, nv_enable_matmul=True, nv_enable_embedding=True, use_apex_cross_entropy=False)
        elif exec == "Thunder-torch.compile":
            executors["Thunder-torch.compile"] = partial(thunderfx, executors=["cudnn","torchcompile"])
        else:
            assert False, f"Unknown executor: {exec}"

    # setup model
    if not args.dtype in torch.__dict__:
        assert False, f"{args.dtype} is not a supported torch data type."
    dtype = torch.__dict__[args.dtype]
    model = module(config)
    if hasattr(config, "lora") and config.lora == True:
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module = patch_linear_module(module, dropout=0.0)
    model = model.cuda().to(dtype)
    if inference:
        model.requires_grad_(False).eval()

    batch_sizes = [config.batch_size]
    if args.batch_sizes is not None:
        batch_sizes = args.batch_sizes
    seq_lens = [config.seq_len]
    if args.seq_lens is not None:
        seq_lens = args.seq_lens
    packed_seq_fn = None
    if args.packed_seqs:
        packed_seq_fn = partial(dummy_packed_seqs, config=config, min_len=2)

    df_tot = None
    for batch_size, seq_len in product(batch_sizes, seq_lens):
        # setup inputs
        local_input_fn = partial(input_fn, dtype, batch_size=batch_size, seq_len=seq_len, packed_seq_fn=packed_seq_fn)
        local_grad_fn = None
        if grad_fn is not None:
            local_grad_fn = partial(grad_fn, dtype, batch_size=batch_size, seq_len=seq_len)

        benchmark_data = []
        for name, exec in executors.items():
            if ("Thunder" in name) or ("torch.compile" in name):
                torch._dynamo.reset()
            if ("nvFuser" in name) or ("default" in name):
                FusionCache.get().reset()
            if (("Thunder" in name) and args.thunder_trace_for_error):
                # Capturing the error prior to runtime execution is using thunder.jit as I am not sure
                # how to do the same with ThunderFX
                exec_model = thunder.jit(model)
                print(thunder.compile_data(exec_model).get_computation_and_inputs(**local_input_fn()))
                exit(0)
            else:
                exec_model = exec(model)

            if (("Thunder" in name) or ("torch.compile" in name)) and (args.dynamo_explain):
                explanation = torch._dynamo.explain(model)(**local_input_fn())
                print(explanation)
 
            if ("Thunder" in name) and args.thunder_trace:
                exec_model(**local_input_fn())
                backend = exec_model._backend
                print(name, "Forward:")
 
                for subgraph_info in backend.subgraph_infos:
                    assert isinstance(subgraph_info.original_graph_module, torch.fx.GraphModule)
                    assert len(subgraph_info.thunder_compiled_fns)  # There was atleast one function compiled with thunder.
                    for thunder_fn in subgraph_info.thunder_compiled_fns:
                        print(thunder.last_traces(thunder_fn)[-1])
 
                if module_has_loss or local_grad_fn is not None:
                    print(name, "Backward:")
                    for subgraph_info in backend.subgraph_infos:
                        assert isinstance(subgraph_info.original_graph_module, torch.fx.GraphModule)
                        assert len(subgraph_info.thunder_compiled_fns)  # There was atleast one function compiled with thunder.
                        for thunder_fn in subgraph_info.thunder_compiled_fns:
                            print(thunder.last_backward_traces(thunder_fn)[-1])
 
            fd_fwd = {}
            fd_bwd = {}
            if (("nvFuser" in name) or (name == "Thunder-default")) and args.nvfuser_repro:
                exec_model(**local_input_fn())
                backend = exec_model._backend
                for subgraph_info in backend.subgraph_infos:
                    assert isinstance(subgraph_info.original_graph_module, torch.fx.GraphModule)
                    assert len(subgraph_info.thunder_compiled_fns)  # There was atleast one function compiled with thunder.
                    for thunder_fn in subgraph_info.thunder_compiled_fns:
                        for key in thunder.last_traces(thunder_fn)[-1].python_ctx().keys():
                            if key[0:8] == "nvFusion":
                                fd_fwd[key] = thunder.last_traces(thunder_fn)[-1].python_ctx()[key]
                                fd_fwd[key].store_inputs = True
 
                if module_has_loss or local_grad_fn is not None:
                    for subgraph_info in backend.subgraph_infos:
                        assert isinstance(subgraph_info.original_graph_module, torch.fx.GraphModule)
                        assert len(subgraph_info.thunder_compiled_fns)  # There was atleast one function compiled with thunder.
                        for thunder_fn in subgraph_info.thunder_compiled_fns:
                            for key in thunder.last_backward_traces(thunder_fn)[-1].python_ctx().keys():
                                if key[0:8] == "nvFusion":
                                    fd_bwd[key] = thunder.last_backward_traces(thunder_fn)[-1].python_ctx()[key]
                                    fd_bwd[key].store_inputs = True
 
            def step_func(input_dict, module_has_loss, grads):
                y = exec_model(**input_dict)
                if module_has_loss or grads is not None:
                    for idx in range(0, len(y)):
                        if idx == 0:
                            loss = y[0]
                        else:
                            loss += y[idx]
 
                    if hasattr(model, "parameters"):
                        for param in model.parameters():
                            param.grad = None
 
                    if module_has_loss:
                        loss.backward()
                    else:
                        loss.backward(grads)
                torch.cuda.synchronize()
 
            def wallclock_time_iter(count):
                input_dict = local_input_fn()
                grads = None
                if local_grad_fn is not None:
                    grads = local_grad_fn()
 
                t = timeit.Timer(
                    stmt="step_func(input_dict, module_has_loss, grads)",
                    globals={"step_func": step_func, "input_dict": input_dict, "module_has_loss": module_has_loss, "grads": grads}
                )
 
                return t.timeit(count)
                
            def kernel_time_iter():
                torch.cuda.nvtx.range_push("Inputs Generation")
                input_dict = local_input_fn()
                torch.cuda.nvtx.range_pop()
                
                torch.cuda.nvtx.range_push("Forward")
                fwd_time = 0.0
                fwd_kernels = 0
                if not args.nsys:
                    with profile(activities=[ProfilerActivity.CUDA]) as prof: 
                        y = exec_model(**input_dict)
                        # NOTE: the THunder-nvFuser executor interfers with CUPTI
                        # collection without putting python to sleep for 5 seconds
                        sleep(5)
 
                    for evt in prof.events():
                        if evt.device_time > 0.0:
                            fwd_kernels += 1
                        fwd_time += evt.device_time
                else:
                    y = exec_model(**input_dict)
                torch.cuda.nvtx.range_pop()
               
                bwd_time = 0.0
                bwd_kernels = 0
                if module_has_loss or local_grad_fn is not None:
                    torch.cuda.nvtx.range_push("Forward-Loss")
                    for idx in range(0, len(y)):
                        if idx == 0:
                            loss = y[0]
                        else:
                            loss += y[idx]
                    torch.cuda.nvtx.range_pop()
 
                    torch.cuda.nvtx.range_push("Grad Generation")
                    if hasattr(model, "parameters"):
                        for param in model.parameters():
                            param.grad = None
                    if not module_has_loss and local_grad_fn is not None:
                        grads = local_grad_fn()
                    torch.cuda.nvtx.range_pop()
 
                    torch.cuda.nvtx.range_push("Backward")
                    if not args.nsys:
                        with profile(activities=[ProfilerActivity.CUDA]) as prof: 
                            if module_has_loss:
                                loss.backward()
                            else:
                                loss.backward(grads)
                            # NOTE: the THunder-nvFuser executor interfers with CUPTI
                            # collection without putting python to sleep for 5 seconds
                            sleep(5)

                        for evt in prof.events():
                            if evt.device_time > 0.0:
                                bwd_kernels += 1
                            bwd_time += evt.device_time
                    else:
                        if module_has_loss:
                            loss.backward()
                        else:
                            loss.backward(grads)
                    torch.cuda.nvtx.range_pop()
 
                return fwd_kernels, fwd_time, bwd_kernels, bwd_time
 
 
            def run_model():
                wallclock_time_iter(args.warmup)
                
                wallclock_time = wallclock_time_iter(args.wallclock_iters) / args.wallclock_iters * 1000.0
               
                fwd_kernel_time = 0
                bwd_kernel_time = 0
                fwd_kernels = 0
                bwd_kernels = 0
                for _ in range(args.cupti_iters):
                    fwd_kernels, fwd_time, bwd_kernels, bwd_time = kernel_time_iter()
                    fwd_kernel_time += fwd_time
                    bwd_kernel_time += bwd_time
 
                fwd_kernel_time = fwd_kernel_time / args.cupti_iters / 1.e3
                bwd_kernel_time = bwd_kernel_time / args.cupti_iters / 1.e3
 
                return fwd_kernels, fwd_kernel_time, bwd_kernels, bwd_kernel_time, wallclock_time
 
            fwd_time = 0.0
            bwd_time = 0.0
            fwd_kernels = 0
            bwd_kernels = 0
            wallclock_time = 0.0
            torch.cuda.nvtx.range_push(f"Executor: {name}")
            try:
                # Locking clocks asssures that kernel times are measured at a consistent
                # clock frequency given that python sleep give enough time for the clocks
                # to be put in power save mode after measuring forward kernels.
                if not args.disable_locked_clocks:
                    lock_max_gpu_clocks()
                fwd_kernels, fwd_time, bwd_kernels, bwd_time, wallclock_time = run_model()
            except Exception as e:
                traceback.print_exc()
                print("Model Exception!", e)
            finally:
                if not args.disable_locked_clocks:
                    reset_gpu_clocks()
            torch.cuda.nvtx.range_pop()
 
            if (("nvFuser" in name) or (name == "Thunder-default")) and args.nvfuser_repro:
                for key in fd_fwd.keys():
                    print(f"nvfuser Forward Repro: {key}")
                    fd_fwd[key].last_used.execute(inputs=fd_fwd[key].last_inputs, print_repro=True)
                for key in fd_bwd.keys():
                    print(f"nvfuser Backward Repro: {key}")
                    fd_bwd[key].last_used.execute(inputs=fd_bwd[key].last_inputs, print_repro=True)
 
            total_time = fwd_time + bwd_time
            #benchmark_data.append([model_name, args.dtype, batch_size, seq_len, fwd_kernels, fwd_time, bwd_kernels, bwd_time, fwd_kernels+bwd_kernels, total_time, wallclock_time, wallclock_time - total_time])
            if inference:
                benchmark_data.append([name, model_name, args.dtype, batch_size, seq_len, fwd_kernels, fwd_time, wallclock_time, wallclock_time - total_time])
            else:
                benchmark_data.append([name, model_name, args.dtype, batch_size, seq_len, fwd_kernels, fwd_time, bwd_kernels, bwd_time, fwd_kernels+bwd_kernels, total_time, wallclock_time, wallclock_time - total_time])

        if inference:
            df = pd.DataFrame(benchmark_data, columns=["Executor", "Model", "DType", "Batch", "Seq-Len", "Fwd-Krnls", "Fwd-K-Time(ms)", "Wall-Time(ms)", "Overhead(ms)"])
            if (len(executors.keys()) > 1) and  ("Torch-Eager" in executors.keys()) :
                df["Fwd-K-Spdup"] = df["Fwd-K-Time(ms)"].rdiv(df.loc[df["Executor"] == "Torch-Eager", 'Fwd-K-Time(ms)'].item())
                df["Wall-Spdup"] = df["Wall-Time(ms)"].rdiv(df.loc[df["Executor"] == "Torch-Eager", 'Wall-Time(ms)'].item())
                new_order = ["Executor", "Model", "DType", "Batch", "Seq-Len", "Fwd-Krnls", "Fwd-K-Time(ms)", "Fwd-K-Spdup", "Wall-Time(ms)", "Wall-Spdup", "Overhead(ms)"]
                df = df[new_order]
        else:
            df = pd.DataFrame(benchmark_data, columns=["Executor", "Model", "DType", "Batch", "Seq-Len", "Fwd-Krnls", "Fwd-K-Time(ms)", "Bwd-Krnls", "Bwd-K-Time(ms)", "Krnls", "K-Time(ms)", "Wall-Time(ms)", "Overhead(ms)"])
            if (len(executors.keys()) > 1) and  ("Torch-Eager" in executors.keys()) :
                df["Fwd-K-Spdup"] = df["Fwd-K-Time(ms)"].rdiv(df.loc[df["Executor"] == "Torch-Eager", 'Fwd-K-Time(ms)'].item())
                df["Bwd-K-Spdup"] = df["Bwd-K-Time(ms)"].rdiv(df.loc[df["Executor"] == "Torch-Eager", 'Bwd-K-Time(ms)'].item())
                df["K-Spdup"] = df["K-Time(ms)"].rdiv(df.loc[df["Executor"] == "Torch-Eager", 'K-Time(ms)'].item())
                df["Wall-Spdup"] = df["Wall-Time(ms)"].rdiv(df.loc[df["Executor"] == "Torch-Eager", 'Wall-Time(ms)'].item())
                new_order = ["Executor", "Model", "DType", "Batch", "Seq-Len", "Fwd-Krnls", "Fwd-K-Time(ms)", "Fwd-K-Spdup", "Bwd-Krnls", "Bwd-K-Time(ms)", "Bwd-K-Spdup", "Krnls", "K-Time(ms)", "K-Spdup", "Wall-Time(ms)", "Wall-Spdup", "Overhead(ms)"]
                df = df[new_order]

        print(df.to_string())

        if df_tot is None:
            df_tot = df
        else:
            df_tot = pd.concat([df_tot, df], ignore_index=True)

    if args.csv:
        print(df_tot.to_csv(model_name + ".csv", index=False))

    if args.pivot:
        df_tot_pivot = None
        if inference:
            df_tot_pivot = df_tot.pivot(index=["Model", "Batch", "Seq-Len"], columns=["Executor"], values=["Fwd-K-Time(ms)", "Fwd-K-Spdup", "Wall-Time(ms)", "Wall-Spdup", "Overhead(ms)"])
        else:
            df_tot_pivot = df_tot.pivot(index=["Model", "Batch", "Seq-Len"], columns=["Executor"], values=["Fwd-K-Time(ms)", "Fwd-K-Spdup", "Bwd-K-Time(ms)", "Bwd-K-Spdup", "K-Time(ms)", "K-Spdup", "Wall-Time(ms)", "Wall-Spdup", "Overhead(ms)"])

        print(df_tot_pivot.to_string())


    return None
