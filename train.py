import numpy as np 
import os 

import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader 
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

from emotic import Emotic 
from emotic_dataset import Emotic_PreDataset
from prepare_models import prep_models
from test import test_data, test_scikit_ap


def train_data(opt, scheduler, models, device, train_loader, val_loader, train_writer, val_writer, model_path, args, ind2cat, ind2vad):
    '''
    Training emotic model on train data using train loader.
    :param opt: Optimizer object.
    :param scheduler: Learning rate scheduler object.
    :param models: Emotic model. 
    :param device: Torch device. Used to send tensors to GPU if available. 
    :param train_loader: Dataloader iterating over train dataset. 
    :param val_loader: Dataloader iterating over validation dataset. 
    :param train_writer: SummaryWriter object to save train logs. 
    :param val_writer: SummaryWriter object to save validation logs. 
    :param model_path: Directory path to save the models after training. 
    :param args: Runtime arguments.
    :param ind2cat: Dictionary converting integer index to categorical emotion.
    :param ind2vad: Dictionary converting integer index to continuous emotion dimension.
    '''

    criterion = nn.BCELoss()
    best_val_ap = 0.0  # 跟踪最佳验证mAP
    
    emotic_model = models

    emotic_model.to(device)

    print ('starting training')

    for e in range(args.epochs):
        running_loss = 0.0 
        running_cat_loss = 0.0 
        running_cont_loss = 0.0
        
        # 用于存储训练阶段的预测和标签
        train_cat_preds = []
        train_cat_labels = []
        
        emotic_model.train()
        
        #train models for one epoch 
        train_iterator = tqdm(train_loader, desc=f'Epoch {e+1}/{args.epochs} [Train]', leave=False)
        for images_context, images_body, labels_cat, labels_cont in train_iterator:
            images_context = images_context.to(device)
            images_body = images_body.to(device)
            labels_cat = labels_cat.to(device)
            labels_cont = labels_cont.to(device)

            opt.zero_grad()

            pred_cat = emotic_model(images_context, images_body)
            loss = criterion(pred_cat, labels_cat)
            
            running_loss += loss.item()
            
            # 收集训练阶段的预测和标签
            train_cat_preds.append(pred_cat.detach().cpu().numpy())
            train_cat_labels.append(labels_cat.detach().cpu().numpy())
            
            loss.backward()
            opt.step()

        # 计算训练阶段的mAP
        train_cat_preds = np.concatenate(train_cat_preds, axis=0).transpose()
        train_cat_labels = np.concatenate(train_cat_labels, axis=0).transpose()
        
        if e % 1 == 0: 
            print ('epoch = %d loss = %.4f cat loss = %.4f cont_loss = %.4f' %(e, running_loss, running_cat_loss, running_cont_loss))
            print('Training metrics:')
            train_ap = test_scikit_ap(train_cat_preds, train_cat_labels, ind2cat, train_writer, e)

        train_writer.add_scalar('losses/total_loss', running_loss, e)
        
        running_loss = 0.0 
        
        # 用于存储验证阶段的预测和标签
        val_cat_preds = []
        val_cat_labels = []
        
        emotic_model.eval()
        
        with torch.no_grad():
            #validation for one epoch
            val_iterator = tqdm(val_loader, desc=f'Epoch {e+1}/{args.epochs} [Val]', leave=False)
            for images_context, images_body, labels_cat, labels_cont in val_iterator:
                images_context = images_context.to(device)
                images_body = images_body.to(device)
                labels_cat = labels_cat.to(device)
                labels_cont = labels_cont.to(device)

                pred_cat = emotic_model(images_context, images_body)
                loss = criterion(pred_cat, labels_cat)
                
                running_loss += loss.item()
                
                # 收集验证阶段的预测和标签
                val_cat_preds.append(pred_cat.cpu().numpy())
                val_cat_labels.append(labels_cat.cpu().numpy())

        # 计算验证阶段的mAP
        val_cat_preds = np.concatenate(val_cat_preds, axis=0).transpose()
        val_cat_labels = np.concatenate(val_cat_labels, axis=0).transpose()

        if e % 1 == 0:
            print ('epoch = %d validation loss = %.4f cat loss = %.4f cont loss = %.4f ' %(e, running_loss, running_cat_loss, running_cont_loss))
            print('Validation metrics:')
            val_ap = test_scikit_ap(val_cat_preds, val_cat_labels, ind2cat, val_writer, e)
            
            # 如果当前验证mAP更好，则保存模型
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                print(f'New best validation mAP: {best_val_ap:.4f}, saving model...')
                emotic_model.to("cpu")
                torch.save(emotic_model, os.path.join(model_path, 'model_emotic1_best.pth'))
                emotic_model.to(device)
            
            # 记录当前学习率
            current_lr = opt.param_groups[0]['lr']
            print(f'Current learning rate: {current_lr}')
            train_writer.add_scalar('learning_rate', current_lr, e)
        
        val_writer.add_scalar('losses/total_loss', running_loss, e)
        
        scheduler.step(running_loss)  # 使用验证loss来调整学习率
    
    print ('completed training')
    print(f'Best validation mAP: {best_val_ap:.4f}')
    print ('saved models')


def train_emotic(result_path, model_path, train_log_path, val_log_path, ind2cat, ind2vad, context_norm, body_norm, args):
    ''' Prepare dataset, dataloders, models. 
    :param result_path: Directory path to save the results (val_predidictions mat object, val_thresholds npy object).
    :param model_path: Directory path to load pretrained base models and save the models after training. 
    :param train_log_path: Directory path to save the training logs. 
    :param val_log_path: Directoty path to save the validation logs. 
    :param ind2cat: Dictionary converting integer index to categorical emotion. 
    :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
    :param context_norm: List containing mean and std values for context images. 
    :param body_norm: List containing mean and std values for body images. 
    :param args: Runtime arguments. 
    '''
    # Load preprocessed data from npy files
    train_context = np.load(os.path.join(args.data_path, 'train_context_arr.npy'))
    train_body = np.load(os.path.join(args.data_path, 'train_body_arr.npy'))
    train_cat = np.load(os.path.join(args.data_path, 'train_cat_arr.npy'))
    train_cont = np.load(os.path.join(args.data_path, 'train_cont_arr.npy'))

    val_context = np.load(os.path.join(args.data_path, 'val_context_arr.npy'))
    val_body = np.load(os.path.join(args.data_path, 'val_body_arr.npy'))
    val_cat = np.load(os.path.join(args.data_path, 'val_cat_arr.npy'))
    val_cont = np.load(os.path.join(args.data_path, 'val_cont_arr.npy'))

    test_context = np.load(os.path.join(args.data_path, 'test_context_arr.npy'))
    test_body = np.load(os.path.join(args.data_path, 'test_body_arr.npy'))
    test_cat = np.load(os.path.join(args.data_path, 'test_cat_arr.npy'))
    test_cont = np.load(os.path.join(args.data_path, 'test_cont_arr.npy'))

    print ('train ', 'context ', train_context.shape, 'body', train_body.shape, 'cat ', train_cat.shape, 'cont', train_cont.shape)
    print ('val ', 'context ', val_context.shape, 'body', val_body.shape, 'cat ', val_cat.shape, 'cont', val_cont.shape)
    print ('test ', 'context ', test_context.shape, 'body', test_body.shape, 'cat ', test_cat.shape, 'cont', test_cont.shape)

    # Initialize Dataset and DataLoader 
    train_transform = transforms.Compose([transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])])

    train_dataset = Emotic_PreDataset(train_context, train_body, train_cat, train_cont, train_transform, context_norm, body_norm)
    val_dataset = Emotic_PreDataset(val_context, val_body, val_cat, val_cont, test_transform, context_norm, body_norm)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)

    print ('train loader ', len(train_loader), 'val loader ', len(val_loader))

    # 创建Emotic模型
    emotic_model = Emotic(512, 512)  # ResNet18的特征维度是512
    
    device = torch.device("cuda:%s" %(str(args.gpu)) if torch.cuda.is_available() else "cpu")

    # 收集需要训练的参数
    param_groups = [
        {
            'params': emotic_model.fusion.parameters(),
            'lr': 0.001
        },
        {
            'params': emotic_model.resnet_body.parameters(),
            'lr': 0.00001
        },
        {
            'params': list(emotic_model.blip_transform.parameters()) + 
                     list(emotic_model.resnet_bbox_transform.parameters()),
            'lr': 0.001
        }
    ]
    opt = optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=5
    )

    train_writer = SummaryWriter(train_log_path)
    val_writer = SummaryWriter(val_log_path)

    # training
    train_data(opt, scheduler, emotic_model, device, train_loader, val_loader, train_writer, val_writer, model_path, args, ind2cat, ind2vad)
    # validation
    test_data(emotic_model, device, val_loader, ind2cat, ind2vad, len(val_dataset), result_dir=result_path, test_type='val', writer=val_writer, epoch=args.epochs)
