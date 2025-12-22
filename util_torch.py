from torch.autograd import Variable
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
# from model import *
import random
import copy
import datetime
import time
import torch
from operator import truediv
import matplotlib.pyplot as plt

def mynorm(data, norm_type):
    data_norm = np.zeros(data.shape)    
    if norm_type == 'bandwise':
        for i in range(data.shape[2]):
            data_max = np.max(data[:,:,i])
            data_min = np.min(data[:,:,i])
            data_norm[:,:,i] = (data[:,:,i]-data_min)/(data_max-data_min)
    elif norm_type == 'pixelwise':
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data_max = np.max(data[i,j,:])
                data_min = np.min(data[i,j,:])
                data_norm[i,j,:] = (data[i,j,:]-data_min)/(data_max-data_min)
    return data_norm

# In[2]  create the image patches
def createPatches(X, y, windowSize, removeZeroLabels=False):
    # X = X.reshape(X.shape[2], X.shape[0], X.shape[1])
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), 'symmetric')
    zeroPaddedX = zeroPaddedX.reshape(zeroPaddedX.shape[2], zeroPaddedX.shape[0], zeroPaddedX.shape[1])
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], X.shape[2], windowSize, windowSize))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for c in range(margin, zeroPaddedX.shape[2] - margin):
        for r in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[:, r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def random_sample(train_sample, validate_sample, patchesLabels):
    num_classes = int(np.max(patchesLabels))
    dataList = patchesLabels
    TrainIndex = []
    TestIndex = []
    ValidateIndex = []

    for i in range(num_classes):
        train_sample_temp = train_sample[i]
        validate_sample_temp = validate_sample[i]
        index = np.where(patchesLabels == (i + 1))[0]
        Train_Validate_Index = random.sample(range(0, int(index.size)), train_sample_temp + validate_sample_temp)
        TrainIndex = np.hstack((TrainIndex, index[Train_Validate_Index[0:train_sample_temp]])).astype(np.int32)
        ValidateIndex = np.hstack((ValidateIndex, index[Train_Validate_Index[train_sample_temp:100000]])).astype(np.int32)
        Test_Index = [index[i] for i in range(0, len(index), 1) if i not in Train_Validate_Index]
        TestIndex = np.hstack((TestIndex, Test_Index)).astype(np.int32)
        
    np.random.shuffle(TrainIndex)
    np.random.shuffle(ValidateIndex)
    np.random.shuffle(TestIndex)

    return TrainIndex, ValidateIndex, TestIndex


# In[3]  apply PCA preprocessing for data sets
def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca

# In[4]: calculate the classification result
# def reports(y_pred, target_1):
#     classification = classification_report(target_1, y_pred)
#     confusion = confusion_matrix(target_1, y_pred)
#     oa = np.trace(confusion) / sum(sum(confusion))
#     ca = np.diag(confusion) / confusion.sum(axis=1)
#     Pe = (confusion.sum(axis=0) @ confusion.sum(axis=1)) / np.square(sum(sum(confusion)))
#     K = (oa - Pe) / (1 - Pe)
#     aa = sum(ca) / len(ca)
#     List = []
#     List.append(np.array(oa)), List.append(np.array(aa)), List.append(np.array(K))
#     List = np.array(List)
#     accuracy_matrix = np.concatenate((ca, List), axis=0)
#     # return classification, confusion, accuracy_matrix
#     return oa, aa, K, ca,  accuracy_matrix



class Multidata(torch.utils.data.Dataset):
    def __init__(self,data1,data2, gt, patch_size):
        super(Multidata, self).__init__()
        self.data1 = data1  
        self.data2 = data2
        self.label = gt-1
        self.patch_size = patch_size
        
        self.data_all_offset1 = np.zeros((data1.shape[0] + self.patch_size - 1, self.data1.shape[1] + self.patch_size - 1, self.data1.shape[2]))
        self.start = int((self.patch_size - 1) / 2)
        self.data_all_offset1[self.start:data1.shape[0] + self.start, self.start:data1.shape[1] + self.start, :] = self.data1[:, :, :]
        
        self.data_all_offset2 = np.zeros((data2.shape[0] + self.patch_size - 1, self.data2.shape[1] + self.patch_size - 1, self.data2.shape[2]))
        self.start = int((self.patch_size - 1) / 2)
        self.data_all_offset2[self.start:data2.shape[0] + self.start, self.start:data2.shape[1] + self.start, :] = self.data2[:, :, :]
        
        x_pos, y_pos = np.nonzero(gt)
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos)])
        self.labels = [self.label[x,y] for x,y in self.indices]
        
#         np.random.shuffle(self.indices)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        x, y = self.indices[i]
#         x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
#         x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data1 = self.data_all_offset1[x:x+self.patch_size, y:y+self.patch_size]
        label = self.label[x, y]
        data1 = np.asarray(data1.transpose((2, 0, 1)), dtype='float32')
        
        data2 = self.data_all_offset2[x:x+self.patch_size, y:y+self.patch_size]
        data2 = np.asarray(data2.transpose((2, 0, 1)), dtype='float32')       
        
        
        label = np.asarray(label, dtype='int64')
        data1 = torch.from_numpy(data1)
        data2 = torch.from_numpy(data2)
        label = torch.from_numpy(label)
#         print(type(data[6,6,1]),data.shape)
#         print(label)
        return data1,data2, label

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def reports(y_pred, target_1):
    classification = classification_report(target_1, y_pred)
    confusion = confusion_matrix(target_1, y_pred)
    oa = np.trace(confusion) / sum(sum(confusion))
    ca = np.diag(confusion) / confusion.sum(axis=1)
    Pe = (confusion.sum(axis=0) @ confusion.sum(axis=1)) / np.square(sum(sum(confusion)))
    K = (oa - Pe) / (1 - Pe)
    aa = sum(ca) / len(ca)
    each_acc, _ = AA_andEachClassAccuracy(confusion)
    List = []
    List.append(np.array(oa)), List.append(np.array(aa)), List.append(np.array(K))
    List = np.array(List)
    accuracy_matrix = np.concatenate((ca, List), axis=0)
    return oa, aa, K, each_acc,  accuracy_matrix

# In[5]: Def val
def val(model, val_loader, criterion):
    global acc, acc_best
    model.eval()
    total_correct = 0
    eye = torch.eye(int(max(val_loader.dataset.labels) + 1)).cuda()
    avg_loss = 0.0
    with torch.no_grad():
        for i, (data_hsi, data_lidar, labels) in enumerate(val_loader):
            data_hsi, data_lidar, labels = Variable(data_hsi).cuda(), Variable(data_lidar).cuda(), Variable(labels).cuda()
            output = model(data_hsi, data_lidar)
            # labels = labels.to(torch.int64)
            # target_hot = eye[labels]
            # avg_loss = criterion(output, target_hot)
            loss = criterion(output, labels.long())
            avg_loss = avg_loss + loss.item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
            acc = float(total_correct) / len(val_loader.dataset)

    avg_loss /= len(val_loader)
    acc = float(total_correct) / len(val_loader.dataset)

    return acc, avg_loss

# In[5]: Def training
def train(model, criterion, device, train_loader, optimizer, scheduler, EPOCHS, val_loader, itera=1):
    global best_model
    acc_temp = 0
    epoch_temp = 1
    eye = torch.eye(int(max(train_loader.dataset.labels) + 1)).cuda()
    start_time_train = datetime.datetime.now()
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    for epoch in range(1, EPOCHS + 1):
        start_time = datetime.datetime.now()
        model.train()
        number = 0
        train_avg_loss = 0
        for batch_idx, (data_hsi, data_lidar, target) in enumerate(train_loader):
            data_hsi, data_lidar, target = data_hsi.to(device), data_lidar.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data_hsi, data_lidar)
            # target = target.to(torch.int64)
            # target_hot = eye[target]
            # loss = criterion(output, target_hot)
            loss = criterion(output, target.long())
            train_avg_loss = train_avg_loss + loss.item()
            loss.backward()
            optimizer.step()
            output = output.argmax(dim=1)
            number += output.eq(target).float().sum().item()
            cur_time = datetime.datetime.now()
            
        train_acc = number/len(train_loader.dataset)
        train_avg_loss /= len(train_loader) 
        val_acc, avg_loss = val(model, val_loader, criterion)
        scheduler.step()
        print('epoch %d, train loss %.6f, train acc %.3f, valida  loss %.6f, valida acc %.3f'% (epoch, train_avg_loss, train_acc, avg_loss, val_acc))

        if acc_temp <= val_acc:
            print('Best_Val_Value changed: from %f to %f;' % (acc_temp, val_acc), end="\t")
            epoch_temp = epoch
            acc_temp = val_acc
            best_model = copy.deepcopy(model)
            print('Best Classification Accuracy %f， Best Classification loss %f； Best Epoch： %d' % (
            acc_temp, avg_loss, epoch_temp), end="\n")
            es = 0
        else:
            if EPOCHS >100:
                if epoch > 100:
                    es += 1
                    print("Counter {} of 20".format(es))
                    if es > 20:
                        print("Early stopping with best_val_acc: ", acc_temp, "at epoch %d: "%(epoch_temp) + "...")
                        break
            else:
                if epoch > 50:
                    es += 1
                    print("Counter {} of 20".format(es))
                    if es > 20:
                        print("Early stopping with best_val_acc: ", acc_temp, "at epoch %d: "%(epoch_temp) + "...")
                        break
        # vis.line(Y=[[number / len(train_loader.dataset), val_acc]],
        #           X=[epoch],
        #           win='acc {}'.format(itera),
        #           opts=dict(title='acc', legend=['acc', 'val_acc']),
        #           update='append')
        
    model = best_model
    end_time_train = datetime.datetime.now()
    print('||======= Train Time for % s' % (end_time_train - start_time_train), '======||')
    return model, (end_time_train - start_time_train).total_seconds()

# In[6]: Def test
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    y_pred = []
    target_1 = []
    # torch.cuda.synchronize()
    start_time_test = datetime.datetime.now()
    with torch.no_grad():
        for (data_hsi, data_lidar, target) in test_loader:
            data_hsi, data_lidar, target = data_hsi.to(device), data_lidar.to(device), target.to(device)
            target = target.to(torch.int64)
            output = model(data_hsi, data_lidar)
            y_pred_temp = output.max(1, keepdim=True)[1]
            correct += y_pred_temp.eq(target.view_as(y_pred_temp)).sum().item()
            y_pred_temp_1 = y_pred_temp.data.cpu().numpy()
            target_temp_1 = target.data.cpu().numpy()
            y_pred.extend(y_pred_temp_1)
            target_1.extend(target_temp_1)

        y_pred = np.array(y_pred)
        y_pred = y_pred.reshape(1, y_pred.size)
        y_pred = np.array(y_pred).astype(np.float32)
        y_pred = y_pred[0]

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc_temp = format(100. * correct / len(test_loader.dataset))
    # test_acc.append(test_acc_temp)
    test_loss_temp = format(test_loss)
    end_time_test = datetime.datetime.now()
    print('||======= Test Time for % s' % (end_time_test - start_time_test), '======||')
    return test_acc_temp, test_loss_temp, y_pred, target_1, (end_time_test - start_time_test).total_seconds()


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 1:
            y[index] = np.array([255, 170, 0]) / 255.
        if item == 2:
            y[index] = np.array([65, 105, 225]) / 255.
        if item == 3:
            y[index] = np.array([190, 255, 232]) / 255.
        if item == 4:
            y[index] = np.array([35, 230, 40]) / 255.
        if item == 5:
            y[index] = np.array([156, 156, 156]) / 255.
        if item == 6:
            y[index] = np.array([115, 0, 0]) / 255.
        if item == 7:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 8:
            y[index] = np.array([0, 168, 132]) / 255.
        if item == 9:
            y[index] = np.array([127, 255, 212]) / 255.
        if item == 10:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 11:
            y[index] = np.array([115, 223, 255]) / 255.
        if item == 12:
            y[index] = np.array([205, 205, 102]) / 255.
        if item == 13:
            y[index] = np.array([137, 90, 68]) / 255.
        if item == 14:
            y[index] = np.array([215, 158, 158]) / 255.
        if item == 15:
            y[index] = np.array([255, 115, 223]) / 255.
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 17:
            y[index] = np.array([76, 0, 115]) / 255.
        if item == 18:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.
    return y

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0


def infer_allmap(model, device, datasetName, FileName, labels, all_loader, itera):
    predicts = np.zeros((0))
    model.eval()
    time_1 = time.time()
    with torch.no_grad():
        for (data_hsi, data_lidar, target) in all_loader:
            data_hsi, data_lidar, target = data_hsi.to(device), data_lidar.to(device), target.to(device)
            target = target.to(torch.int64)
            output = model(data_hsi, data_lidar)     
            _, predict = torch.max(output.data, 1)
            predict = predict.cpu().numpy()
            predicts = np.append(predicts, predict)   
    time_use = time.time() - time_1
    
    gt = labels.flatten()
    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 17
    gt = gt[:] - 1
    
    y_all = list_to_colormap(predicts)
    y_gt = list_to_colormap(gt)

    gt_re = np.reshape(y_gt, (labels.shape[0], labels.shape[1], 3))
    y_all_map = np.reshape(y_all, (labels.shape[0], labels.shape[1], 3))

    classification_map(gt_re, labels, 300,
                       './' + FileName  + '/' + 'classification' + '_'+ FileName + '_' + datasetName  + '_gt.png')

    classification_map(y_all_map, labels, 300,
                        './' + FileName  + '/' + 'classification'+ '_' + str(itera) + '_'+ FileName + '_' + datasetName + '_all_map.png')
    
    return time_use




