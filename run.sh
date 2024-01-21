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
               --gt_type_train 0\
               --loop 0 \
            #    --phase test\