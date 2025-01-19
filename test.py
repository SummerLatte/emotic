import numpy as np 
import os 
import scipy.io
from sklearn.metrics import average_precision_score, precision_recall_curve

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm

from emotic import Emotic 
from emotic_dataset import Emotic_PreDataset


def test_scikit_ap(cat_preds, cat_labels, ind2cat, writer=None, epoch=None):
  ''' Calculate average precision per emotion category using sklearn library.
  :param cat_preds: Categorical emotion predictions. 
  :param cat_labels: Categorical emotion labels. 
  :param ind2cat: Dictionary converting integer index to categorical emotion.
  :param writer: TensorBoard writer object.
  :param epoch: Current epoch number.
  :return: Numpy array containing average precision per emotion category.
  '''
  ap = np.zeros(26, dtype=np.float32)
  for i in range(26):
    ap[i] = average_precision_score(cat_labels[i, :], cat_preds[i, :])
    print ('Category %16s %.5f' %(ind2cat[i], ap[i]))
    if writer is not None and epoch is not None:
      writer.add_scalar(f'metrics/AP_{ind2cat[i]}', ap[i], epoch)
      
  mean_ap = ap.mean()
  print ('Mean AP %.5f' %(mean_ap))
  if writer is not None and epoch is not None:
    writer.add_scalar('metrics/mAP', mean_ap, epoch)
  return ap


def test_vad(cont_preds, cont_labels, ind2vad):
  ''' Calcaulate VAD (valence, arousal, dominance) errors. 
  :param cont_preds: Continuous emotion predictions. 
  :param cont_labels: Continuous emotion labels. 
  :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
  :return: Numpy array containing mean absolute error per continuous emotion dimension. 
  '''
  vad = np.zeros(3, dtype=np.float32)
  for i in range(3):
    vad[i] = np.mean(np.abs(cont_preds[i, :] - cont_labels[i, :]))
    print ('Continuous %10s %.5f' %(ind2vad[i], vad[i]))
  print ('Mean VAD Error %.5f' %(vad.mean()))
  return vad


def get_thresholds(cat_preds, cat_labels):
  ''' Calculate thresholds where precision is equal to recall. These thresholds are then later for inference.
  :param cat_preds: Categorical emotion predictions. 
  :param cat_labels: Categorical emotion labels. 
  :return: Numpy array containing thresholds per emotion category where precision is equal to recall.
  '''
  thresholds = np.zeros(26, dtype=np.float32)
  for i in range(26):
    p, r, t = precision_recall_curve(cat_labels[i, :], cat_preds[i, :])
    for k in range(len(p)):
      if p[k] == r[k]:
        thresholds[i] = t[k]
        break
  return thresholds


def test_data(models, device, test_loader, ind2cat, ind2vad, num_images, result_dir=None, test_type=None, writer=None, epoch=None):
    '''
    Function to test models on test data.
    :param models: Emotic model.
    :param device: Torch device. Used to send tensors to GPU if available.
    :param test_loader: Dataloader iterating over test dataset. 
    :param ind2cat: Dictionary converting integer index to categorical emotion. 
    :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
    :param num_images: Number of images being tested.
    :param result_dir: Directory path to save the results (model predictions).
    :param test_type: Test type variable. Variable used in directory path to save results.
    :param writer: SummaryWriter object to save the logs in tensorboard (or None).
    :param epoch: Current epoch number (or None).
    '''
    with torch.no_grad():
        emotic_model = models
        emotic_model.to(device)
        emotic_model.eval()
        
        cat_preds = []
        cat_labels = []
        
        test_iterator = tqdm(test_loader, desc='Testing', leave=False)
        for images_context, images_body, images_face, has_face, labels_cat, labels_cont in test_iterator:
            images_context = images_context.to(device)
            images_body = images_body.to(device)
            images_face = images_face.to(device)
            has_face = has_face.to(device)
            labels_cat = labels_cat.to(device)
            labels_cont = labels_cont.to(device)

            pred_cat = emotic_model(images_context, images_body, images_face, has_face)
            
            cat_preds.append(pred_cat.cpu().numpy())
            cat_labels.append(labels_cat.cpu().numpy())
        
        cat_preds = np.concatenate(cat_preds, axis=0).transpose()
        cat_labels = np.concatenate(cat_labels, axis=0).transpose()
        
        # 计算并打印测试指标
        test_scikit_ap(cat_preds, cat_labels, ind2cat, writer, epoch)
        
        # 保存预测结果
        if result_dir and test_type:
            np.save(os.path.join(result_dir, '{}_cat_preds.npy'.format(test_type)), cat_preds)
            np.save(os.path.join(result_dir, '{}_cat_labels.npy'.format(test_type)), cat_labels)
            print ('Saved results in directory ', result_dir)


def test_emotic(result_path, model_path, ind2cat, ind2vad, context_norm, body_norm, args):
    ''' Prepare test data and test models on the same.
    :param result_path: Directory path to save the results (val_predidictions mat object, val_thresholds npy object).
    :param model_path: Directory path to load pretrained base models and save the models after training. 
    :param ind2cat: Dictionary converting integer index to categorical emotion. 
    :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
    :param context_norm: List containing mean and std values for context images. 
    :param body_norm: List containing mean and std values for body images. 
    :param args: Runtime arguments.
    '''    
    # Prepare models 
    # model_context = torch.load(os.path.join(model_path,'model_context1.pth'))
    # model_body = torch.load(os.path.join(model_path,'model_body1.pth'))
    emotic_model = torch.load(os.path.join(model_path,'model_emotic1_best.pth'))
    print ('Succesfully loaded models')

    #Load data preprocessed npy files
    test_context = np.load(os.path.join(args.data_path, 'test_context_arr.npy'))
    test_body = np.load(os.path.join(args.data_path, 'test_body_arr.npy'))
    test_face = np.load(os.path.join(args.data_path, 'test_face_arr.npy'))
    test_has_face = np.load(os.path.join(args.data_path, 'test_has_face.npy'))
    test_cat = np.load(os.path.join(args.data_path, 'test_cat_arr.npy'))
    test_cont = np.load(os.path.join(args.data_path, 'test_cont_arr.npy'))
    print ('test ', 'context ', test_context.shape, 'body', test_body.shape, 'cat ', test_cat.shape, 'cont', test_cont.shape)

    # Initialize Dataset and DataLoader 
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])])
    test_dataset = Emotic_PreDataset(test_context, test_body, test_face, test_has_face, test_cat, test_cont, test_transform, context_norm, body_norm)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
    print ('test loader ', len(test_loader))
    
    device = torch.device("cuda:%s" %(str(args.gpu)) if torch.cuda.is_available() else "cpu")
    test_data(emotic_model, device, test_loader, ind2cat, ind2vad, len(test_dataset), result_dir=result_path, test_type='test', writer=None, epoch=None)
