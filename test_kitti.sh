# python mono/tools/test_scale_cano.py \
#     'mono/configs/HourglassDecoder/test_kitti_convlarge_hourglass_0.3_150.py' \
#     --load-from ./weight/convlarge_hourglass_0.3_150_step750k_v1.1.pth \
#     --test_data_path /root/code/Metric3D/data/EMDB/P0_00_mvs_a.json \
#     --show-dir /root/code/Metric3D/results/EMDB/ \
#     --launcher None
python launch.py
cd /root/code/detection && sh train.sh