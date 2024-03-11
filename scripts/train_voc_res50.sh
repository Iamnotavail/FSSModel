python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --node_rank=0 --master_port=16008 \
./train.py --datapath "../dataset" \
           --benchmark pascal \
           --fold 3 \
           --bsz 8 \
           --nworker 4 \
           --backbone resnet50 \
           --feature_extractor_path "../backbone/resnet50_a1h-35c100f8.pth" \
           --logpath "./logs" \
           --lr 0.001 \
           --nepoch 100; \
