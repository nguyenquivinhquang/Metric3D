import os
from datetime import datetime
import subprocess
import multiprocessing as mp
from concurrent import futures
import shutil
import argparse
DATASET_DIR = '/root/code/Metric3D/data/EMDB/'
# OUTPUT_DIR = "/root/code/multi-hmr/results/"

def run(gpus, seq):
    cur_proc = mp.current_process()
    # print(cur_proc)
    worker_id = cur_proc._identity[0] - 1  # 1-indexed processes

    log_file = seq.split('/')[-1]
    log_file = f"logs/{log_file}.log"
    os.makedirs("logs", exist_ok=True)
    cmd = f"python mono/tools/test_scale_cano.py 'mono/configs/HourglassDecoder/test_kitti_convlarge_hourglass_0.3_150.py' \
    --load-from ./weight/convlarge_hourglass_0.3_150_step750k_v1.1.pth \
    --test_data_path {seq} \
    --show-dir /root/code/Metric3D/results/EMDB/ \
    --launcher None"
    # cmd = f"CUDA_VISIBLE_DEVICES={gpu} python demo.py --data {video_name} --disable_vis  --output_dir {save_dir} --intrinsic_params {intrinsic[0]} {intrinsic[1]} {intrinsic[2]} {intrinsic[3]}"
    print("Process", seq)
    cmd = f"{cmd} > {log_file} 2>&1"
        
    subprocess.call(cmd, shell=True)
    print("Done", seq)
    return
if __name__ == "__main__":
    # seqs_P = ['P4', 'P5', 'P6']
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="*", default=[0])
    parser.add_argument("--num_threads",  type=int, default=10)

    args = parser.parse_args()
    seqs = []

    for file in sorted(os.listdir(DATASET_DIR)):
        seqs.append(
            os.path.join(DATASET_DIR, file)
        )
    with futures.ProcessPoolExecutor(max_workers=args.num_threads) as exe:
        for i, seq in enumerate(seqs):
            
            exe.submit(
                run,
                args.gpus,
                seq
            )
    # run(args.gpus, seqs[0])

