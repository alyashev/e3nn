import argparse
import logging
import os
import pandas
import sys
import time

import torch
import intel_extension_for_pytorch as ipex

from e3nn.o3 import Irreps, FullyConnectedTensorProduct
from e3nn.util.jit import compile


logging.basicConfig(level=logging.DEBUG)

def get_data(profiler, table, shapes=False):
    profiler_data = table.splitlines()
    profiler_data = profiler_data[3:-3]
    profiler_data = [line for line in profiler_data if '---' not in line]
    profiler_data = [line.replace('autograd::engine::evaluate_function: ', '') for line in profiler_data]
    profiler_data = [line.replace('torch::autograd...', '') for line in profiler_data]
    profiler_data = [line.replace('torch::autograd::', '') for line in profiler_data]
    profiler_data = [line.replace('<forward op>', '<forward-op>') for line in profiler_data]
    profiler_data = [line.replace('torch::jit::', '') for line in profiler_data]
    profiler_data = [line.replace('(anonymous namespace)::', '') for line in profiler_data]
    #profiler_data = [line.replace('anonymous', '') for line in profiler_data]
    profiler_data = [line.replace('<backward op>','<backward-op>') for line in profiler_data]

    profiler_data = [line.replace(', ', ',') for line in profiler_data]
    profiler_data = [line.split() for line in profiler_data]

    columns = ['name', 'self_cpu_perc', 'self_cpu',
           'cpu_total_perc', 'cpu_total', 'cpu_time_avg',
           'self_xpu', 'self_xpu_perc',
           'xpu_total', 'xpu_time_avg',
           'nr_of_calls']
    if shapes:
        columns += ['input_shapes']

    output = pandas.DataFrame(profiler_data, columns=columns)
    output = output.set_index(columns[0])

    return output
    
crt_dir = os.path.dirname(os.path.realpath(__file__))

# https://stackoverflow.com/a/15008806/1008938
def t_or_f(arg) -> bool:
    ua = str(arg).upper()
    if "TRUE".startswith(ua):
        return True
    elif "FALSE".startswith(ua):
        return False
    else:
        raise ValueError(str(arg))


def main() -> None:
    parser = argparse.ArgumentParser(prog="tensor_product_benchmark")
    parser.add_argument("--jit", type=t_or_f, default=True)
    parser.add_argument("--irreps-in1", type=str, default="8x0e + 8x1e + 8x2e + 8x3e")
    parser.add_argument("--irreps-in2", type=str, default="8x0e + 8x1e + 8x2e + 8x3e")
    parser.add_argument("--irreps-out", type=str, default="8x0e + 8x1e + 8x2e + 8x3e")
    parser.add_argument("--cuda", type=t_or_f, default=True)
    parser.add_argument("--backward", type=t_or_f, default=True)
    parser.add_argument("--opt-ein", type=t_or_f, default=True)
    parser.add_argument("--specialized-code", type=t_or_f, default=True)
    parser.add_argument("-w", type=int, default=10)
    parser.add_argument("-n", type=int, default=3)
    parser.add_argument("--batch", type=int, default=10)

    args = parser.parse_args()

    device = "xpu" #"cuda" if (torch.cuda.is_available() and args.cuda) else "cpu"
    args.cuda = device == "cuda"

    if args.cuda:
        # Workaround for CUDA driver issues
        # See https://github.com/pytorch/pytorch/issues/60158#issuecomment-866294291
        with torch.profiler.profile() as _:
            pass

    print("======= Benchmark with settings: ======")
    for key, val in vars(args).items():
        print(f"{key:>18} : {val}")
    print("=" * 40)

    irreps_in1 = Irreps(args.irreps_in1)
    irreps_in2 = Irreps(args.irreps_in2)
    irreps_out = Irreps(args.irreps_out)
    tp = FullyConnectedTensorProduct(
        irreps_in1, irreps_in2, irreps_out, _specialized_code=args.specialized_code, _optimize_einsums=args.opt_ein
    )
    tp = tp.to(device=device)

    inputs = [
        (irreps_in1.randn(args.batch, -1).to(device=device), irreps_in2.randn(args.batch, -1).to(device=device))
        for _ in range(1 + args.w + args.n)
    ]
    if args.backward:
        for tmp in inputs:
            for t in tmp:
                t.requires_grad_(True)
    inputs = iter(inputs)

    # compile
    if args.jit:
        print("JITing...")
        tp = compile(tp)

    ipex.optimize(tp)

    print("starting...")

    called_num = [0]

    def trace_handler(p) -> None:
        print(p.key_averages().table(sort_by="self_xpu_time_total", row_limit=-1))
        p.export_chrome_trace("test_trace_" + str(called_num[0]) + ".json")
        called_num[0] += 1

    show_shapes = False
    with torch.autograd.profiler_legacy.profile(use_xpu=True, record_shapes=show_shapes
        #activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #activities=[torch.profiler.ProfilerActivity.XPU],
        #schedule=torch.profiler.schedule(wait=1, warmup=args.w, active=args.n),
        #on_trace_ready=trace_handler,
    ) as p:
        for _ in range(1 + args.w + args.n):
            out = tp(*next(inputs))
            if args.backward:
                # tanh() forces it to realize the grad as a full size matrix rather than expanded (stride 0) ones
                out.tanh().sum().backward()
            #p.step()
    #trace_handler(p)
    profile_table= p.key_averages().table(sort_by="self_xpu_time_total")
    profile_data = get_data(p, profile_table, show_shapes)
    profile_data_filename = os.path.join(crt_dir, 'tensor_product_profile.csv')
        #f'{args.model}.b{batch_size:03}.s{sequence_length:04}.p{padding:04}.t{max_new_tokens:04}.profile.csv')
    profile_data.to_csv(profile_data_filename)


if __name__ == "__main__":
    main()
