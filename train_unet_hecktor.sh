declare -a gt_types=(-1)

for i in "${gt_types[@]}"
do
   echo "Annotator $i"
   python main.py --dataset Hecktor \
               --net_arch Unet \
               --random_seed 27\
               --batch_size 32\
               --num_worker 8 \
               --learning_rate 5e-5 \
               --weight_decay 0.0 \
               --num_epoch 200 \
               --rank 1 \
               --dataroot /home/kudaibergen/projects/sa/data/hecktor22_test_2d\
               --use_non_empty\
               --phase train\
               --gt_type_train $i\
               --loop 1\
               --notes ''
done