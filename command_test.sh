# CUDA_VISIBLE_DEVICES=0 python test.py --camera kinect --dump_dir logs/log_kn4/dump_epoch10 --checkpoint_path logs/log_kn4/minkuresunet_epoch10.tar --batch_size 1 --dataset_root /home/yzhang/graspness_data/graspnet --infer --eval --collision_thresh -1
CUDA_VISIBLE_DEVICES=1 python test.py --camera realsense --dump_dir logs/log_realsense/dump_epoch10 --checkpoint_path logs/log_realsense/minkuresunet_epoch10.tar --batch_size 1 --dataset_root /home/yzhang/graspness_data/graspnet --infer --eval --collision_thresh -1