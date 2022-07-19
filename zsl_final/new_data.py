from os.path import join
from scipy import io
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import build_model
from resnet_feature import resnet101_features
import pickle
from torchvision import transforms
import random

torch.backends.cudnn.deterministic = True
seed = 7384
print("Random Seed: ", seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

feat_mat = io.loadmat("/HDD-1_data/arlen/zsl_final/datasets/Data/SUN/res101.mat")
att_mat = io.loadmat("/HDD-1_data/arlen/zsl_final/datasets/Data/SUN/att_splits.mat")

imgroot = "/HDD-1_data/arlen/dataset"
img_files = np.squeeze(feat_mat['image_files'])
new_img_files = []
for img_file in img_files:
    img_path = img_file[0]
    img_path = join(imgroot+'/SUN', '/'.join(img_path.split('/')[7:]))
    new_img_files.append(img_path)

new_img_files = np.array(new_img_files)
device = "cuda" if torch.cuda.is_available() else "cpu"

res101 = resnet101_features(pretrained=True)
w2v_file = "SUN_attribute.pkl"
w2v_path = join("/HDD-1_data/arlen/zsl_final/datasets/Attribute/w2v/", w2v_file)
with open(w2v_path, 'rb') as f:
    w2v = pickle.load(f)
w2v = torch.from_numpy(w2v).float().to(device)
model = build_model.DSACA_Net(res101, w2v).to(device)

model.load_state_dict(torch.load("/HDD-1_data/arlen/zsl_final/log/model/SUN_348_24_7384.pth"))
model.eval()

tfs = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


trainvalloc = att_mat['trainval_loc'].squeeze() - 1
train_img = new_img_files[trainvalloc]
print(train_img[6856])
img = Image.open(train_img[6856]).convert('RGB')
img = tfs(img)
img = img.unsqueeze(0).to(device)
new_feature = model(img, mode="gan")
print(new_feature)

new_feat_mat = np.zeros((1, 2048))
for img_path in tqdm(new_img_files):
    img = Image.open(img_path).convert('RGB')
    img = tfs(img)
    img = img.unsqueeze(0).to(device)
    new_feature = model(img, mode="gan")
    new_feat_mat = np.vstack((new_feat_mat, new_feature.cpu().detach().numpy()))
new_feat_mat = new_feat_mat[1:]
train_feat = new_feat_mat[trainvalloc]
print(train_feat[6856])
io.savemat("new_data/SUN_feat.mat", mdict={'features': new_feat_mat})

my_mat = io.loadmat("/HDD-1_data/arlen/zsl_final/new_data/" + "SUN" + "_feat.mat")
my_feature = my_mat['features']
my_train_feat = my_feature[trainvalloc]
print(my_train_feat[6856])
