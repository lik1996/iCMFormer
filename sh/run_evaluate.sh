MODEL_PATH=./weights/icmformer/icmformer_cocolvis_vit_base.pth

python scripts/evaluate_model.py NoBRS \
--gpu=0 \
--checkpoint=${MODEL_PATH} \
--eval-mode=cvpr \
--datasets=GrabCut