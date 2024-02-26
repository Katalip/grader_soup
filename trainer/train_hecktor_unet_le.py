import os
import cv2
import torch
import numpy as np
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from utils.metrics import get_soft_dice


gt_type_train_to_annotator = {
    -1: 'Majority Voting',
    0: 'A1',
    1: 'A2',
    2: 'A3',
    3: 'A4',
}

AUX_WEIGHT = 0.3

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))


def save_checkpoint(model, optimizer, save_path):
    checkpoint = {
            'model': model.state_dict()
        }
    torch.save(checkpoint, save_path)


def train_hecktor_le(args, log_folder, checkpoint_folder, visualization_folder, metrics_folder,
                   model, optimizer, loss_function, train_set, gt_train_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    model = model.cuda()
    train_loader = DataLoader(train_set, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              num_workers=args.num_worker, 
                              pin_memory=True)
    rmse_loss = RMSELoss()
    amp_grad_scaler = GradScaler()

    n_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable params:', (n_trainable_parameters / 1.0e+6), 'M')
    experiment_name = f'{gt_train_name}_loop{args.loop}'
    print(f'Experiment: {experiment_name}')

    wandb.init(
        entity='katalip',
        dir=log_folder, 
        project='super_annot', 
        config=vars(args), 
        notes=experiment_name)

    for this_epoch in tqdm(range(args.num_epoch)):
        print(this_epoch)
        model.train()
        train_loss = 0.0
        train_var_loss = 0.0
        train_soft_dice_gtvp = 0.0
        train_soft_dice_gtvn = 0.0

        for step, data in enumerate(tqdm(train_loader, total=len(train_loader))):
            imgs = data['image'].to(dtype=torch.float32).cuda()
            mask = data['mask']
            mask_major_vote = data['majority_mask']

            optimizer.zero_grad()
            with autocast():
                outputs = model({'image': imgs})

            gt_mask = mask_major_vote.cuda()
            
            total_loss = 0
            outputs_sigmoid = []
            weights = [AUX_WEIGHT if i < len(outputs) - 1 else 1 for i in range(len(outputs))]
            for i, out in enumerate(outputs):
                layer_gt_mask = gt_mask
                if args.use_label_sampling:
                    if i < len(outputs) - 1:
                        mask_index = np.random.randint(0, len(mask))
                        layer_gt_mask = mask[mask_index].cuda()
                

                out = torch.nn.functional.interpolate(out, size=layer_gt_mask.shape[2:])
                outputs_sigmoid.append(torch.sigmoid(out))
                loss_value = loss_function(out, layer_gt_mask)
                total_loss += weights[i] * loss_value
            
            preds = torch.stack(outputs_sigmoid, dim=0).sum(dim=0) / len(outputs)

            if args.use_var_loss:
                variance_heatmap_output = torch.stack(outputs_sigmoid, dim=0).var(dim=0)
                variance_heatmap_mask = torch.stack(mask, dim=0).var(dim=0)
                var_loss = rmse_loss(variance_heatmap_output, variance_heatmap_mask.half().cuda())
                total_loss += 5*var_loss
                train_var_loss += 5*var_loss * imgs.size(0)

            amp_grad_scaler.scale(total_loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()

            train_loss += (total_loss.item() * imgs.size(0))
            train_soft_dice_gtvn = train_soft_dice_gtvn + get_soft_dice(outputs=preds[:, 1, :, :].cpu(),
                                                                      masks=gt_mask[:, 1, :, :].cpu()) * imgs.size(0)
            train_soft_dice_gtvp = train_soft_dice_gtvp + get_soft_dice(outputs=preds[:, 0, :, :].cpu(),
                                                                        masks=gt_mask[:, 0, :, :].cpu()) * imgs.size(0)
        wandb.log({"Loss/train": train_loss / train_set.__len__()})
        wandb.log({"Train/dice_gtvp": train_soft_dice_gtvp / train_set.__len__()})
        wandb.log({"Train/dice_gtvn": train_soft_dice_gtvn / train_set.__len__()})

        if args.use_var_loss:
            wandb.log({"Train/var_loss": train_var_loss / train_set.__len__()})

        save_checkpoint(model, optimizer, checkpoint_folder + '/last.pt')


def validate_hecktor_le(args, model, val_set, loss_func):
    model = model.cuda()
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True)

    model.eval()
    val_loss = 0.0
    val_soft_dice_gtvn = 0.0
    val_soft_dice_gtvp = 0.0

    for step, data in enumerate(tqdm(val_loader, total=len(val_loader))):
        with torch.no_grad():
            imgs = data['image'].to(dtype=torch.float32).cuda()
            mask = data['mask']
            mask_major_vote = data['majority_mask']

            output = model({'image': imgs})

            if args.gt_type_train == -1:
                gt_mask = mask_major_vote.cuda()
            else:
                gt_mask = mask[args.gt_type_train].cuda()
            
            loss = loss_func(output['raw'], gt_mask)
            preds = torch.sigmoid(output['raw'])

            val_loss += (loss.item() * imgs.size(0))
            val_soft_dice_gtvp += get_soft_dice(
                outputs=preds[:, 0, :, :].cpu(),
                masks=gt_mask[:, 0, :, :].cpu()) * imgs.size(0)
            val_soft_dice_gtvn += get_soft_dice(
                outputs=preds[:, 1, :, :].cpu(), 
                masks=gt_mask[:, 1, :, :].cpu()) * imgs.size(0)
            
    return (val_loss / val_set.__len__(), 
            val_soft_dice_gtvp / val_set.__len__(), 
            val_soft_dice_gtvn / val_set.__len__())


def test_visualization(imgs_name, Preds_visual, visualization_folder):
    no_samples = Preds_visual.size(0)
    print(visualization_folder)
    for idx in range(no_samples):
        Pred = np.uint8(Preds_visual[idx].detach().cpu().numpy() * 255)
        Pred_gtvp = cv2.applyColorMap(Pred[0], cv2.COLORMAP_JET)
        Pred_gtvn = cv2.applyColorMap(Pred[1], cv2.COLORMAP_JET)
        Pred_path = visualization_folder + '/' + imgs_name[idx][15:-4] + '_Pred_TAB.png'
        print(Pred_path)
        os.makedirs(os.path.dirname(Pred_path), exist_ok=True)
        cv2.imwrite(Pred_path, 0.5 * Pred_gtvp + 0.5 * Pred_gtvn)


def test_hecktor_le(args, visualization_folder, metrics_folder, model, test_set):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.cuda()
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True)

    metrix_file = metrics_folder + "/dice.txt"
    file_handle = open(metrix_file, 'a')
    if args.validate:
        file_handle.write('validation data size: %d \n' % (test_set.__len__()))
    else:
        file_handle.write('testing data size: %d \n' % (test_set.__len__()))
    file_handle.close()

    model.eval()
    test_soft_dice_gtvn = 0.0
    test_soft_dice_gtvp = 0.0

    test_soft_dice_gtvp_raters = [0.0] * args.rater_num
    test_soft_dice_gtvn_raters = [0.0] * args.rater_num

    imgs_visual = list()
    Preds_visual = list()

    for step, data in enumerate(test_loader):
        with torch.no_grad():
            imgs = data['image'].to(dtype=torch.float32).cuda()
            mask = data['mask']
            mask_major_vote = data['majority_mask']

            outputs = model({'image': imgs})
            mask_major_vote = mask_major_vote.cuda()

            outputs_sigmoid = []
            for out in outputs:
                out = torch.nn.functional.interpolate(out, size=mask_major_vote.shape[2:])
                outputs_sigmoid.append(torch.sigmoid(out))
            
            preds = torch.stack(outputs_sigmoid, dim=0).sum(dim=0) / len(outputs)

            imgs_visual.extend(data['name'])
            Preds_visual.append(preds)

            test_soft_dice_gtvn += get_soft_dice(
                outputs=preds[:, 1, :, :].cpu(), 
                masks=mask_major_vote[:, 1, :, :].cpu()) * imgs.size(0)
            
            test_soft_dice_gtvp += get_soft_dice(
                outputs=preds[:, 0, :, :].cpu(),
                masks=mask_major_vote[:, 0, :, :].cpu()) * imgs.size(0)

            test_soft_dice_gtvp_raters = [
                test_soft_dice_gtvp_raters[i] + get_soft_dice(outputs=preds[:, 0, :, :].cpu(),
                                                              masks=mask[i][:, 0, :, :].cpu()) * imgs.size(0) for i in range(len(mask))]
            test_soft_dice_gtvn_raters = [
                test_soft_dice_gtvn_raters[i] + get_soft_dice(outputs=preds[:, 1, :, :].cpu(),
                                                             masks=mask[i][:, 1, :, :].cpu()) * imgs.size(0) for i in range(len(mask))]

    Preds_visual = torch.cat(Preds_visual, dim=0)
    test_visualization(imgs_visual, Preds_visual, visualization_folder)

    file_handle = open(metrix_file, 'a')
    file_handle.write("Mean Voting: ({}, {})\n".format(round(test_soft_dice_gtvp / test_set.__len__() * 100, 2),
                                                       round(test_soft_dice_gtvn / test_set.__len__() * 100, 2)))
    file_handle.write(
        "Average: ({}, {})\n".format(round(np.mean(test_soft_dice_gtvp_raters) / test_set.__len__() * 100, 2),
                                     round(np.mean(test_soft_dice_gtvn_raters) / test_set.__len__() * 100, 2)))

    for i in range(len(mask)):
        file_handle.write(
            "rater{}: ({}, {})\n".format(i + 1, round(test_soft_dice_gtvp_raters[i] / test_set.__len__() * 100, 2),
                                         round(test_soft_dice_gtvn_raters[i] / test_set.__len__() * 100, 2)))


def get_label_pred_list(args, bs, masks, output_sample_list):
    gtvn_label_list = []
    gtvp_label_list = []
    for idx in range(bs):
        temp_gtvn_label_list = []
        temp_gtvp_label_list = []
        for anno_no in range(args.rater_num):
            temp_gtvn_label = masks[anno_no][idx, 1, :, :].to(dtype=torch.float32)
            temp_gtvp_label = masks[anno_no][idx, 0, :, :].to(dtype=torch.float32)
            temp_gtvn_label_list.append(temp_gtvn_label)
            temp_gtvp_label_list.append(temp_gtvp_label)
        gtvn_label_list.append(temp_gtvn_label_list)
        gtvp_label_list.append(temp_gtvp_label_list)

    gtvn_pred_list = []
    gtvp_pred_list = []
    for idx in range(bs):
        temp_gtvn_pred_list = []
        temp_gtvp_pred_list = []
        for pred_no in range(len(output_sample_list)):
            temp_gtvn_pred = torch.sigmoid(output_sample_list[pred_no][idx, 1, :, :])
            temp_gtvp_pred = torch.sigmoid(output_sample_list[pred_no][idx, 0, :, :])
            temp_gtvn_pred_list.append(temp_gtvn_pred)
            temp_gtvp_pred_list.append(temp_gtvp_pred)
        gtvn_pred_list.append(temp_gtvn_pred_list)
        gtvp_pred_list.append(temp_gtvp_pred_list)
    return gtvn_label_list, gtvp_label_list, gtvn_pred_list, gtvp_pred_list
