outputs="./outputs/base"

python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 eval.py --config configs/cfg_voc20.py  --work-dir $outputs --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 eval.py --config configs/cfg_voc21.py  --work-dir $outputs --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 eval.py --config configs/cfg_ade20k.py  --work-dir $outputs --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 eval.py --config configs/cfg_city_scapes.py  --work-dir $outputs --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 eval.py --config configs/cfg_context59.py  --work-dir $outputs --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 eval.py --config configs/cfg_context60.py  --work-dir $outputs --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 eval.py --config configs/cfg_coco_object.py  --work-dir $outputs --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 eval.py --config configs/cfg_coco_stuff164k.py  --work-dir $outputs --launcher pytorch


cd $outputs
find . -type f -name "*.log" | while read logfile
do
    grep "data_root =" "$logfile"
    grep "dataset_type =" "$logfile"
    grep -o "mIoU: [0-9.]*" "$logfile"
    echo ""
done