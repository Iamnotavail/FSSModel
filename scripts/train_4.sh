python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --node_rank=0 --master_port=16006 \
./train.py --datapath "../dataset" \
           --benchmark pascal \
           --fold 0 \
           --bsz 8 \
           --nworker 4 \
           --backbone swin \
           --feature_extractor_path "../backbone/swin_base_patch4_window12_384_22kto1k.pth" \
           --logpath "./logs" \
           --lr 0.001 \
           --nepoch 100; \
python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --node_rank=0 --master_port=16007 \
./train.py --datapath "../dataset" \
           --benchmark pascal \
           --fold 1 \
           --bsz 8 \
           --nworker 4 \
           --backbone swin \
           --feature_extractor_path "../backbone/swin_base_patch4_window12_384_22kto1k.pth" \
           --logpath "./logs" \
           --lr 0.001 \
           --nepoch 100; \
python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --node_rank=0 --master_port=16008 \
./train.py --datapath "../dataset" \
           --benchmark pascal \
           --fold 2 \
           --bsz 8 \
           --nworker 4 \
           --backbone swin \
           --feature_extractor_path "../backbone/swin_base_patch4_window12_384_22kto1k.pth" \
           --logpath "./logs" \
           --lr 0.001 \
           --nepoch 100; \
python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --node_rank=0 --master_port=16009 \
./train.py --datapath "../dataset" \
           --benchmark pascal \
           --fold 3 \
           --bsz 8 \
           --nworker 4 \
           --backbone swin \
           --feature_extractor_path "../backbone/swin_base_patch4_window12_384_22kto1k.pth" \
           --logpath "./logs" \
           --lr 0.001 \
           --nepoch 100; \



