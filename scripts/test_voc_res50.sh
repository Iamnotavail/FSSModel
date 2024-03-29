python ./test.py --datapath "../dataset" \
                 --benchmark pascal \
                 --fold 0 \
                 --bsz 1 \
                 --nworker 0 \
                 --backbone resnet50 \
                 --feature_extractor_path "../backbone/resnet50_a1h-35c100f8.pth" \
                 --logpath "./logs" \
                 --load "./logs/train/fold_0_0830_131202/best_model.pt" \
                 --nshot 5 \
                 --vispath "./vis_5/" \
