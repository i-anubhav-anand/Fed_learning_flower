import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets ,transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools

import tqdm as tqdm

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import cv2
from PIL import Image 
from torch.autograd import Variable

import sys,getopt
 
import covid




TEST_DATA_PATH = 'dataset\dataset'

test_transforms = transforms.Compose([
                                      transforms.Resize((150,150)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


test_image = datasets.ImageFolder(TEST_DATA_PATH, transform=test_transforms)

testloader = torch.utils.data.DataLoader(test_image, batch_size=1)




if torch.cuda.is_available():  #PyTorch has device object to load the data into the either of two hardware [CPU or CUDA(GPU)]
    DEVICE=torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")
    

    

# model = covid.Net.CNN_Model(pretrained=True)
inf_model = covid.Net.CNN_Model(pretrained=False)
inf_model.to(torch.device('cpu'))
inf_model.load_state_dict(torch.load('best_model _kag.pth', map_location='cpu'))
inf_model.eval()

# specify loss function (categorical cross-entropy loss)
criterion = nn.CrossEntropyLoss() 

# Specify optimizer which performs Gradient Descent
optimizer = optim.Adam(inf_model.parameters(), lr=0.01)


# base_model = covid.train(inf_model, testloader, epochs=1, device=DEVICE)

class_names = testloader.dataset.classes

def singlr_pred(img_path):
    # img_path ='dataset/dataset/normal/IM-0191-0001.jpeg'

    image = cv2.imread(img_path)
    image = cv2.resize(image, (150,150))
    
    image_tensor = test_transforms(Image.fromarray(image))
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    
    input = input.to(torch.device('cpu'))
    out = inf_model(input)
    class_prob = torch.softmax(out, dim=1)

    # get most probable class and its probability:
    class_prob, topclass = torch.max(class_prob, dim=1)

    _, preds = torch.max(out, 1)
    idx = preds.cpu().numpy()[0]
    pred_class = class_names[idx]
    score = out[0][0].item()
    plt.imshow(np.array(image))
    print("Predicted: {}".format(pred_class))
    print("Probability: {}".format(class_prob.item()))

def batch_pred():

    print('Loading.....')

    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for x_batch, y_batch in tqdm.tqdm(testloader, leave=False):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            y_test_pred = inf_model(x_batch)
            y_test_pred = torch.log_softmax(y_test_pred, dim=1)
            _, y_pred_tag = torch.max(y_test_pred, dim = 1)
            y_pred_list.append(y_pred_tag.cpu().numpy())
            y_true_list.append(y_batch.cpu().numpy())
            
    y_pred_list = [i[0] for i in y_pred_list]
    y_true_list = [i[0] for i in y_true_list]

    print(classification_report(y_true_list, y_pred_list))

    cm =  confusion_matrix(y_true_list, y_pred_list)

    plot_confusion_matrix(cm = cm, 
                        normalize    = False,
                        target_names = ['covid19','normal'],
                        title        = "Confusion Matrix")



def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def main(argv):

    param_1= sys.argv[1] 
    param_2= sys.argv[2] 
    if param_1 == '1':
        singlr_pred(param_2)
    else: 
        batch_pred()


    
if __name__ == "__main__":
    main(sys.argv[1:])






