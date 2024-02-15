declare -a gt_types=(-1)

for i in "${gt_types[@]}"
do
   echo "Annotator $i"
   python main.py --dataset RIGA \
               --net_arch UnetLE \
               --random_seed 27\
               --batch_size 16\
               --num_worker 8 \
               --learning_rate 5e-5 \
               --weight_decay 0.0 \
               --num_epoch 200 \
               --rank 1 \
               --dataroot /media/kudaibergen/TS512/projects_ts/grader_soup/data/DiscRegion/DiscRegion\
               --standardize\
               --validate\
               --gt_type_train $i\
               --loop 6\
               --notes LayerEns_weighted0.3_var_loss\
               --use_var_loss
done