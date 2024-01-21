declare -a gt_indices=(0 1 2 3 4 5)

for i in "${gt_indices[@]}"
do
   for j in "${gt_indices[@]}"
      do
         if [ "$i" -lt "$j" ]; then
            echo "Annotators $i $j"
            python main.py --dataset RIGA \
                        --net_arch Unet \
                        --random_seed 27\
                        --batch_size 16\
                        --num_worker 8 \
                        --learning_rate 5e-5 \
                        --weight_decay 0.0 \
                        --num_epoch 200 \
                        --rank 1 \
                        --net_arch Unet\
                        --dataroot /media/kudaibergen/TS512/projects_ts/grader_soup/data/DiscRegion/DiscRegion\
                        --standardize\
                        --validate\
                        --use_mix_label\
                        --mix_label_type union\
                        --gt_index_1 $i\
                        --gt_index_2 $j\
                        --loop 0
         fi
      done
done