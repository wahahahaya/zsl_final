import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_feature import resnet101_features


class DSACAN(nn.Module):
    def __init__(self, res101, w2v):
        super(DSACAN, self).__init__()

        self.w2v_att = w2v
        self.backbone = res101

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
        self.U = nn.Parameter(nn.init.normal_(torch.empty(256, 312)), requires_grad=True)
        self.V = nn.Parameter(nn.init.normal_(torch.empty(256, 49)), requires_grad=True)
        self.atten2attri = nn.Linear(312, 312)

        # util
        self.softmax = nn.Softmax(dim=-1)
        self.feat2attri = nn.Parameter(nn.init.normal_(torch.empty(2048, 312)), requires_grad=True)

    def base_module(self, feat, seen_att):
        # N, C, W, H = feat.shape
        # global_feat = F.avg_pool2d(feat, kernel_size=(W, H))
        # global_feat = global_feat.view(N, C)
        # gs_feat = torch.einsum('bc,cd->bd', global_feat, self.feat2attri)

        gs_feat_norm = torch.norm(feat, p=2, dim=1).unsqueeze(1).expand_as(feat)
        gs_feat_normalized = feat.div(gs_feat_norm + 1e-5)

        temp_norm = torch.norm(seen_att, p=2, dim=1).unsqueeze(1).expand_as(seen_att)
        seen_att_normalized = seen_att.div(temp_norm + 1e-5)

        cos_dist = torch.einsum('bd,nd->bn', gs_feat_normalized, seen_att_normalized)
        score = cos_dist * 20.0

        return score

    def attention_module(self, img):
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

        mfb_R = torch.einsum("tl,bln->btn", self.U.T, img_L)  # [B, 312, 49]
        mfb_Q = self.V.T@text_L  # [49, 312]
        mfb_F = (mfb_R*mfb_Q.T).view(B, 312, W, H)  # [B, 312, 7, 7]

        attention = F.avg_pool2d(mfb_F, kernel_size=(W, H)).view(B, -1)  # [B, 312]
        attribute = self.atten2attri(attention)

        return mfb_F, attribute

    def forward(self, x, attri):
        feat = self.backbone(x)  # [B, 2048, 7, 7]
        mfb_F, attention_out = self.attention_module(feat)
        pred_class = self.base_module(attention_out, attri)

        return attention_out, pred_class
