import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet_feature import resnet101_features
# import GEMZSL.modeling.utils as utils
from numpy import genfromtxt


def get_attributes_info():
    info = {
        "input_dim": 312,
        "n": 200,
        "m": 50
    }

    return info


def get_attr_group():
    attr_group = {
        1: [i for i in range(0, 9)],
        2: [i for i in range(9, 24)],
        3: [i for i in range(24, 39)],
        4: [i for i in range(39, 54)],
        5: [i for i in range(54, 58)],
        6: [i for i in range(58, 73)],
        7: [i for i in range(73, 79)],
        8: [i for i in range(79, 94)],
        9: [i for i in range(94, 105)],
        10: [i for i in range(105, 120)],
        11: [i for i in range(120, 135)],
        12: [i for i in range(135, 149)],
        13: [i for i in range(149, 152)],
        14: [i for i in range(152, 167)],
        15: [i for i in range(167, 182)],
        16: [i for i in range(182, 197)],
        17: [i for i in range(197, 212)],
        18: [i for i in range(212, 217)],
        19: [i for i in range(217, 222)],
        20: [i for i in range(222, 236)],
        21: [i for i in range(236, 240)],
        22: [i for i in range(240, 244)],
        23: [i for i in range(244, 248)],
        24: [i for i in range(248, 263)],
        25: [i for i in range(263, 278)],
        26: [i for i in range(278, 293)],
        27: [i for i in range(293, 308)],
        28: [i for i in range(308, 312)],
    }

    return attr_group


class GEMNet(nn.Module):
    def __init__(self, res101, img_size, c, w, h,
                 attritube_num, cls_num, ucls_num, attr_group, w2v,
                 scale=20.0, device=None):

        super(GEMNet, self).__init__()
        self.device = device

        self.img_size = img_size
        # self.prototype_shape = prototype_shape
        self.attritube_num = attritube_num

        self.feat_channel = c
        self.feat_w = w
        self.feat_h = h

        self.ucls_num = ucls_num
        self.scls_num = cls_num - ucls_num
        self.attr_group = attr_group

        self.w2v_att = torch.from_numpy(w2v).float().to(self.device)  # 312 * 300
        assert self.w2v_att.shape[0] == self.attritube_num

        if scale <= 0:
            self.scale = nn.Parameter(torch.ones(1) * 20.0)
        else:
            self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)

        self.backbone = res101

        # self.prototype_vectors = nn.Parameter(nn.init.normal_(torch.empty(self.prototype_shape)), requires_grad=True)  # a, c

        self.W = nn.Parameter(nn.init.normal_(torch.empty(self.w2v_att.shape[1], self.feat_channel)),
                              requires_grad=True)  # 300 * 2048

        self.V = nn.Parameter(nn.init.normal_(torch.empty(self.feat_channel, self.attritube_num)), requires_grad=True)

        # loss
        self.Reg_loss = nn.MSELoss()
        self.CLS_loss = nn.CrossEntropyLoss()

        # DSACA
        # img attention
        self.query_conv = nn.Conv2d(2048, 256, kernel_size=(1, 1))
        self.key_conv = nn.Conv2d(2048, 256, kernel_size=(1, 1))
        self.value_conv = nn.Conv2d(2048, 2048, kernel_size=(1, 1))
        self.gamma_img = nn.Parameter(torch.zeros(1))

        # text attention
        self.query_W = nn.Linear(300, 256)
        self.key_W = nn.Linear(300, 256)
        self.value_W = nn.Linear(300, 300)
        self.gamma_text = nn.Parameter(torch.zeros(1))

        # co-attention
        self.img2L = nn.Conv2d(2048, 256, kernel_size=(1, 1))
        self.text2L = nn.Linear(300, 256)
        self.relu = nn.ReLU()
        self.UU = nn.Parameter(nn.init.normal_(torch.empty(256, 312)), requires_grad=True)
        self.VV = nn.Parameter(nn.init.normal_(torch.empty(256, 49)), requires_grad=True)
        self.atten2attri = nn.Linear(312, 312)

        # util
        self.softmax = nn.Softmax(dim=-1)
        self.feat2attri = nn.Parameter(nn.init.normal_(torch.empty(2048, 312)), requires_grad=True)

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.backbone(x)
        return x

    def base_module(self, x, seen_att):
        N, C, W, H = x.shape
        global_feat = F.avg_pool2d(x, kernel_size=(W, H))
        global_feat = global_feat.view(N, C)
        gs_feat = torch.einsum('bc,cd->bd', global_feat, self.V)

        gs_feat_norm = torch.norm(gs_feat, p=2, dim=1).unsqueeze(1).expand_as(gs_feat)
        gs_feat_normalized = gs_feat.div(gs_feat_norm + 1e-5)

        temp_norm = torch.norm(seen_att, p=2, dim=1).unsqueeze(1).expand_as(seen_att)
        seen_att_normalized = seen_att.div(temp_norm + 1e-5)

        cos_dist = torch.einsum('bd,nd->bn', gs_feat_normalized, seen_att_normalized)
        score = cos_dist * self.scale

        return score

    def DSACAN(self, img):
        text = self.w2v_att

        # image self attention
        B, C, W, H = img.shape
        img_query = self.query_conv(img).view(B, 256, W*H).permute(0, 2, 1)  # [B, 49, 256]
        img_key = self.key_conv(img).view(B, 256, W*H)  # [B, 256, 49]
        img_value = self.value_conv(img).view(B, C, W*H)  # [B, 2048, 49]

        img_prob = self.softmax(torch.einsum("bnc,bcm->bnm", img_query, img_key))  # [B, 49, 49]
        img_attention = torch.einsum('bcn,bnn->bcn', img_value, img_prob).view(B, C, W, H)  # [B, 2048, 7, 7]
        img_attention = self.gamma_img*img_attention + img

        # text self attention
        text_query = self.query_W(text)  # [312, 256]
        text_key = self.key_W(text).T  # [256, 312]
        text_value = self.value_W(text)  # [312, 300]

        text_prob = self.softmax(text_query@text_key)  # [312, 312]
        text_attention = text_prob@text_value  # [312, 300]
        text_attention = self.gamma_text*text_attention + text

        # co-attention
        img_L = self.relu(self.img2L(img_attention)).view(B, 256, W*H)  # [B, 256, 49]
        text_L = self.relu(self.text2L(text_attention)).T  # [256, 312]

        mfb_R = torch.einsum("tl,bln->btn", self.UU.T, img_L)  # [B, 312, 49]
        mfb_Q = self.VV.T@text_L  # [49, 312]
        mfb_F = (mfb_R*mfb_Q.T)  # [B, 312, 7, 7]
        atten_map = F.softmax(mfb_F, -1).view(B, 312, W, H)

        attention = F.avg_pool2d(atten_map, kernel_size=(W, H)).view(B, -1)  # [B, 312]
        atten_attr = self.atten2attri(attention)

        query = text_attention
        part_feat = 0

        return part_feat, atten_map, atten_attr, query

    def attentionModule(self, x):

        N, C, W, H = x.shape
        x = x.reshape(N, C, W * H)  # N, V, r=WH

        query = torch.einsum('lw,wv->lv', self.w2v_att, self.W)  # L * V

        atten_map = torch.einsum('lv,bvr->blr', query, x)  # batch * L * r

        atten_map = F.softmax(atten_map, -1)

        x = x.transpose(2, 1)  # batch, WH=r, V
        part_feat = torch.einsum('blr,brv->blv', atten_map, x)  # batch * L * V
        part_feat = F.normalize(part_feat, dim=-1)

        atten_map = atten_map.view(N, -1, W, H)
        atten_attr = F.max_pool2d(atten_map, kernel_size=(W, H))
        atten_attr = atten_attr.view(N, -1)

        return part_feat, atten_map, atten_attr, query

    def attr_decorrelation(self, query):

        loss_sum = 0

        for key in self.attr_group:
            group = self.attr_group[key]
            proto_each_group = query[group]  # g1 * v
            channel_l2_norm = torch.norm(proto_each_group, p=2, dim=0)
            loss_sum += channel_l2_norm.mean()

        loss_sum = loss_sum.float()/len(self.attr_group)

        return loss_sum

    def CPT(self, atten_map):
        """
        :param atten_map: N, L, W, H
        :return:
        """

        N, L, W, H = atten_map.shape
        xp = torch.tensor(list(range(W))).long().unsqueeze(1).to(self.device)
        yp = torch.tensor(list(range(H))).long().unsqueeze(0).to(self.device)

        xp = xp.repeat(1, H)
        yp = yp.repeat(W, 1)

        atten_map_t = atten_map.view(N, L, -1)
        value, idx = atten_map_t.max(dim=-1)

        tx = torch.div(idx, H, rounding_mode='floor')
        ty = idx - H * tx

        xp = xp.unsqueeze(0).unsqueeze(0)
        yp = yp.unsqueeze(0).unsqueeze(0)
        tx = tx.unsqueeze(-1).unsqueeze(-1)
        ty = ty.unsqueeze(-1).unsqueeze(-1)

        pos = (xp - tx) ** 2 + (yp - ty) ** 2

        loss = atten_map * pos

        loss = loss.reshape(N, -1).mean(-1)
        loss = loss.mean()

        return loss

    def forward(self, x, att=None, label=None, seen_att=None):

        feat = self.conv_features(x)  # N， 2048， 14， 14

        score = self.base_module(feat, seen_att)  # N, d
        if not self.training:
            return score

        part_feat, atten_map, atten_attr, query = self.DSACAN(feat)

        Lcls = self.CLS_loss(score, label)
        Lreg = self.Reg_loss(atten_attr, att)

        if self.attr_group is not None:
            Lad = self.attr_decorrelation(query)
        else:
            Lad = torch.tensor(0).float().to(self.device)

        Lcpt = self.CPT(atten_map)
        scale = self.scale.item()

        loss_dict = {
            'Reg_loss': Lreg,
            'Cls_loss': Lcls,
            'AD_loss': Lad,
            'CPT_loss': Lcpt,
            'scale': scale
        }

        return loss_dict


def build_GEMNet():
    info = get_attributes_info()
    attritube_num = info["input_dim"]  # 312
    cls_num = info["n"]  # 200
    ucls_num = info["m"]  # 50

    attr_group = get_attr_group()

    img_size = 224

    # res101 feature size
    c, w, h = 2048, img_size//32, img_size//32

    scale = 20.0

    res101 = resnet101_features(pretrained=True)

    data_set_path = '../../dataset/'
    glove_path = data_set_path + "glove_embedding.csv"
    w2v = genfromtxt(glove_path, delimiter=',', skip_header=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    return GEMNet(res101=res101, img_size=img_size,
                  c=c, w=w, h=h, scale=scale,
                  attritube_num=attritube_num,
                  attr_group=attr_group, w2v=w2v,
                  cls_num=cls_num, ucls_num=ucls_num,
                  device=device)
