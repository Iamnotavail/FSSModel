python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --node_rank=0 --master_port=16005 \
./train.py --datapath "../dataset" \
           --benchmark pascal \
           --fold 3 \
           --bsz 8 \
           --nworker 4 \
           --backbone swin \
           --feature_extractor_path "../backbone/swin_base_patch4_window12_384_22kto1k.pth" \
           --logpath "./logs" \
           --lr 0.001 \
           --nepoch 100 \
