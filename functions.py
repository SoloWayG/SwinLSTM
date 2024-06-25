import numpy as np
import torch
from torch.cuda import amp
from torch.cuda.amp import autocast as autocast
from utils import compute_metrics, visualize
from torch.utils.tensorboard import SummaryWriter
scaler = amp.GradScaler()
writer = SummaryWriter(f'writer/lstm')


def model_forward_single_layer(model, inputs, targets_len, num_layers):
    outputs = []
    states = [None] * len(num_layers)

    inputs_len = inputs.shape[1]
    
    last_input = inputs[:, -1]

    for i in range(inputs_len - 1):
        output, states = model(inputs[:, i], states)
        outputs.append(output)

    for i in range(targets_len):
        output, states = model(last_input, states)
        outputs.append(output)
        last_input = output

    return outputs


def model_forward_multi_layer(model, inputs, targets_len, num_layers):
    states_down = [None] * len(num_layers)
    states_up = [None] * len(num_layers)

    outputs = []

    inputs_len = inputs.shape[1]

    last_input = inputs[:, -1]

    for i in range(inputs_len - 1):
        output, states_down, states_up = model(inputs[:, i], states_down, states_up)
        outputs.append(output)

    for i in range(targets_len):
        output, states_down, states_up = model(last_input, states_down, states_up)
        outputs.append(output)
        last_input = output

    return outputs


def train(args, logger, epoch, model, train_loader, criterion, optimizer):
    model.train()
    num_batches = len(train_loader)
    losses = []

    for batch_idx, batch in enumerate(train_loader):
        inputs, targets,x_d,y_d = batch
        optimizer.zero_grad()
        inputs = inputs.permute(0, 2, 1, 3, 4)
        targets = targets.permute(0, 2, 1, 3, 4)

        inputs, targets = map(lambda x: x.float().to(args.device), [inputs, targets])
        targets_len = targets.shape[1]
        with autocast():
            if args.model == 'SwinLSTM-B':
                outputs = model_forward_single_layer(model, inputs, targets_len, args.depths)

            if args.model == 'SwinLSTM-D':
                outputs = model_forward_multi_layer(model, inputs, targets_len, args.depths_down)

            outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()
            targets_ = torch.cat((inputs[:, 1:], targets), dim=1)
            loss = criterion(outputs, targets_)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        writer.add_scalar('Loss_train',loss.item(),(epoch*train_loader.__len__())+batch_idx)
        losses.append(loss.item())

        if batch_idx and batch_idx % args.log_train == 0:
            logger.info(f'EP:{epoch:04d} BI:{batch_idx:03d}/{num_batches:03d} Loss:{np.mean(losses):.6f}')

    return np.mean(losses)


def test(args, logger, epoch, model, test_loader, criterion, cache_dir):
    model.eval()
    num_batches = len(test_loader)
    losses, mses, ssims = [], [], []

    for batch_idx, batch in enumerate(test_loader):
        inputs, targets,x_d,y_d = batch
        with torch.no_grad():
            inputs, targets = map(lambda x: x.float().to(args.device), [inputs, targets])
            
            inputs = inputs.permute(0, 2, 1, 3, 4)
            targets = targets.permute(0, 2, 1, 3, 4)
            targets_len = targets.shape[1]
            if args.model == 'SwinLSTM-B':
                outputs = model_forward_single_layer(model, inputs, targets_len, args.depths)

            if args.model == 'SwinLSTM-D':
                outputs = model_forward_multi_layer(model, inputs, targets_len, args.depths_down)

            outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()
            targets_ = torch.cat((inputs[:, 1:], targets), dim=1)
            
            losses.append(criterion(outputs, targets_).item())
            writer.add_scalar('Loss_tesst',losses[-1],(epoch*test_loader.__len__())+batch_idx)
            inputs_len = inputs.shape[1]
            outputs = outputs[:, inputs_len - 1:]

            mse, ssim = compute_metrics(outputs, targets)
            writer.add_images('Ground_truth', np.expand_dims(targets.squeeze(2).detach().cpu().numpy()[0],axis=1), 0)
            writer.add_images('Predicts', np.expand_dims(outputs.squeeze(2).detach().cpu().numpy()[0],axis=1), 0)
            mses.append(mse)
            ssims.append(ssim)
            writer.add_scalar('mses',mse,(epoch*test_loader.__len__())+batch_idx)
            writer.add_scalar('ssims',ssim,(epoch*test_loader.__len__())+batch_idx)
            if batch_idx and batch_idx % args.log_valid == 0:
                logger.info(
                    f'EP:{epoch:04d} BI:{batch_idx:03d}/{num_batches:03d} Loss:{np.mean(losses):.6f} MSE:{mse:.4f} SSIM:{ssim:.4f}')
                visualize(inputs, targets, outputs, epoch, batch_idx, cache_dir)

    return np.mean(losses), np.mean(mses), np.mean(ssims)
