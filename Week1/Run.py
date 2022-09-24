import numpy as np
import random
import os
import torch
import wandb

from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score
from preprocessing_dataset import train_set, test_set

from simple_cnn import SimpleCNN

# Training settings
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'torch.cuda.is_available() == True --> device : {device}')

batch_size = 64
epochs = 5
lr = 3e-5
gamma = 0.7
seed = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

# Model load & summary
model = SimpleCNN().to(device)

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=True)

print(f'train dataset size : {len(train_set)} / test dataset size : {len(test_set)}')

# loss function, optimizer, scheduler definition
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

# model trainer
def save_checkpoint(model, filename):
    torch.save(model, filename)

best_accuracy = 0
best_epoch = 0

for epoch in range(epochs):
    epoch_accuracy = 0
    epoch_loss = 0
    wandb.init(project="homework1", entity="choo")

    with tqdm(train_loader, unit='batch') as train_epoch:
        print('-' * 91)
        for data, label in train_epoch:
            train_epoch.set_description(f'[Train] Epoch: {epoch + 1}/{epochs}')

            data = data.to(device)
            label = label.type(torch.LongTensor)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

            label = label.cpu().detach().numpy()
            output = output.argmax(dim=1).cpu().detach().numpy()

            train_epoch.set_postfix(loss=epoch_loss.item(), accuracy=epoch_accuracy.item())

        wandb.log({'train_accuracy': epoch_accuracy, 'train_loss': loss})

        if epoch_accuracy >= best_accuracy:
            prev_model_path = f'Result/[ep{best_epoch}_{best_accuracy:0.6f}].pth'
            if os.path.exists(prev_model_path): os.remove(prev_model_path)

            best_accuracy = epoch_accuracy
            best_epoch = epoch + 1

            save_checkpoint(model, f'Result/[ep{best_epoch}_{best_accuracy:0.6f}].pth')

print(f'best model Info --> epoch: {best_epoch} | accuracy: {best_accuracy:0.6f}')
print('-' * 91)

model = torch.load(f'Result/[ep{best_epoch}_{best_accuracy:0.6f}].pth').to(device)

with torch.no_grad():
    model.eval()
    model_test_accuracy = 0
    model_test_loss = 0

    pred_list = []
    label_list = []

    for data, label in tqdm(test_loader, desc='model test'):
        data = data.to(device)
        label = label.type(torch.LongTensor)
        label = label.to(device)
        test_output = model(data)

        test_loss = criterion(test_output, label)

        acc = (test_output.argmax(dim=1) == label).float().mean()
        model_test_accuracy += acc / len(test_loader)
        model_test_loss += test_loss / len(test_loader)

        test_output = test_output.argmax(dim=1).cpu().numpy()
        label = label.cpu().numpy()
        pred_list.append(test_output)
        label_list.append(label)

precision = precision_score(label_list, pred_list, average='macro')
recall = recall_score(label_list, pred_list, average='macro')
f1 = f1_score(label_list, pred_list, average='macro')
accuracy = model_test_accuracy

print(f'precision_score: {precision:0.6f} | recall_score: {recall:0.6f} | f1_score: {f1:0.6f} --> accuracy: {accuracy:0.6f}')