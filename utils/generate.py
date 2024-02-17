import os

def generate_output_folder(args):

    if args.use_mix_label:
        gt_train_name = f'{args.mix_label_type}_A{args.gt_index_1 + 1}_A{args.gt_index_2 + 1}'
    else:        
        if args.gt_type_train == -1:
            gt_train_name = 'majority_vote'
        else:
            gt_train_name = f'A{args.gt_type_train + 1}'

    output_folder_name = args.output_dir + '/' + args.dataset + "_gt_annots_train_" + str(
        gt_train_name) + "_" + args.net_arch + "_" + args.notes + "_" + args.loss_func + "_pretrain" + str(
        args.pretrained) + "_validate" + str(args.validate)
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    setting_folder_name = output_folder_name + '/bs' + str(args.batch_size) + '_lr' + repr(
        args.learning_rate) + '_wd' + repr(args.weight_decay) + '_epoch' + str(args.num_epoch)
    if not os.path.exists(setting_folder_name):
        os.makedirs(setting_folder_name)

    final_folder_name = setting_folder_name + '/loop' + str(args.loop)
    print(final_folder_name)
    if not os.path.exists(final_folder_name):
        os.makedirs(final_folder_name)

    log_folder = final_folder_name + '/logs'
    checkpoint_folder = final_folder_name + '/checkpoints'
    visualization_folder = final_folder_name + '/visualization'
    metrics_folder = final_folder_name + '/metrics'

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    if not os.path.exists(visualization_folder):
        os.makedirs(visualization_folder)
    if not os.path.exists(metrics_folder):
        os.makedirs(metrics_folder)

    return log_folder, checkpoint_folder, visualization_folder, metrics_folder, gt_train_name
