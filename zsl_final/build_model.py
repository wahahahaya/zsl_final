import torch
import torch.nn as nn
import torch.nn.functional as F


class DSACA_Net(nn.Module):
    def __init__(self, res101, w2v, scale=20.0):
        super(DSACA_Net, self).__init__()

        self.w2v_att = w2v
        self.attritube_num = self.w2v_att.shape[0]
        self.attribute_embed = self.w2v_att.shape[1]
        self.backbone = res101
        self.V = nn.Parameter(nn.init.normal_(torch.empty(2048, self.attritube_num)), requires_grad=True)
        self.LinearV = nn.Linear(2048, 200)

        if scale <= 0:
            self.scale = nn.Parameter(torch.ones(1) * 20.0)
        else:
            self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)

        # DSACA
        # img attention
        self.query_conv = nn.Conv2d(2048, 256, kernel_size=(1, 1))
        self.key_conv = nn.Conv2d(2048, 256, kernel_size=(1, 1))
        self.value_conv = nn.Conv2d(2048, 2048, kernel_size=(1, 1))
        self.gamma_img = nn.Parameter(torch.zeros(1))

        # text attention
        self.query_W = nn.Linear(self.attribute_embed, 256)
        self.key_W = nn.Linear(self.attribute_embed, 256)
        self.value_W = nn.Linear(self.attribute_embed, self.attribute_embed)
        self.gamma_text = nn.Parameter(torch.zeros(1))

        # co-attention
        self.img2L = nn.Conv2d(2048, 256, kernel_size=(1, 1))
        self.text2L = nn.Linear(self.attribute_embed, 256)
        self.relu = nn.ReLU()
        self.UU = nn.Parameter(nn.init.normal_(torch.empty(256, self.attritube_num)), requires_grad=True)
        self.VV = nn.Parameter(nn.init.normal_(torch.empty(256, 196)), requires_grad=True)
        self.atten2attri = nn.Linear(self.attritube_num, self.attritube_num)

        # util
        self.softmax = nn.Softmax(dim=-1)
        self.feat2attri = nn.Parameter(nn.init.normal_(torch.empty(2048, self.attritube_num)), requires_grad=True)

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.backbone(x)
        return x

    def base_module(self, x, seen_att):
        gs_feat = x@self.V

        gs_feat_norm = torch.norm(gs_feat, p=2, dim=1).unsqueeze(1).expand_as(gs_feat)
        gs_feat_normalized = gs_feat.div(gs_feat_norm + 1e-5)

        temp_norm = torch.norm(seen_att, p=2, dim=1).unsqueeze(1).expand_as(seen_att)
        seen_att_normalized = seen_att.div(temp_norm + 1e-5)

        cos_dist = gs_feat_normalized @ seen_att_normalized.T
        score = cos_dist * self.scale
        # score = self.LinearV(global_feat)

        return score

    def base_module_dot(self, x, seen_att):
        gs_feat = x@self.V

        cos_dist = gs_feat @ seen_att.T
        score = cos_dist * self.scale
        # score = self.LinearV(global_feat)

        return score

    def DSACA(self, img_feature):
        text = self.w2v_att

        # image self attention
        B, C, W, H = img_feature.shape
        img_query = self.query_conv(img_feature).view(B, 256, W*H).permute(0, 2, 1)  # [B, 196, 256]
        img_key = self.key_conv(img_feature).view(B, 256, W*H)  # [B, 256, 196]
        img_value = self.value_conv(img_feature).view(B, C, W*H)  # [B, 2048, 196]

        img_prob = self.softmax(torch.einsum("bnc,bcm->bnm", img_query, img_key))  # [B, 196, 196]
        img_attention = torch.einsum('bcn,bnn->bcn', img_value, img_prob).view(B, C, W, H)  # [B, 2048, 14, 14]
        img_attention = self.gamma_img*img_attention + img_feature

        # text self attention
        text_query = self.query_W(text)  # [312, 256]
        text_key = self.key_W(text).T  # [256, 312]
        text_value = self.value_W(text)  # [312, 300]

        text_prob = self.softmax(text_query@text_key)  # [312, 312]
        text_attention = text_prob@text_value  # [312, 300]
        text_attention = self.gamma_text*text_attention + text

        # co-attention
        img_L = self.relu(self.img2L(img_attention)).view(B, 256, W*H)  # [B, 256, 196]
        text_L = self.relu(self.text2L(text_attention)).T  # [256, 312]

        mfb_R = torch.einsum("tl,bln->btn", self.UU.T, img_L)  # [B, 312, 196]
        mfb_Q = self.VV.T@text_L  # [196, 312]
        mfb_F = (mfb_R*mfb_Q.T)  # [B, 312, 196]
        atten_map = F.softmax(mfb_F, -1).view(B, self.attritube_num, W, H)  # [B, 312, 14, 14]

        attention = F.avg_pool2d(mfb_F.view(B, self.attritube_num, W, H), kernel_size=(W, H)).view(B, -1)  # [B, 312]
        atten_attr = self.atten2attri(attention)

        query = img_attention
        part_feat = mfb_F

        return part_feat, atten_map, atten_attr, query

    def forward(self, x, demo_feat=None, seen_att=None, mode="train"):
        feat = self.conv_features(x)  # [B, 2048, 14, 14]
        N, C, W, H = feat.shape
        global_feat = F.avg_pool2d(feat, kernel_size=(W, H))
        global_feat = global_feat.view(N, C)
        if mode == "gan":
            return global_feat

        if mode == "demo":
            score = self.base_module(demo_feat, seen_att)
            return score

        score = self.base_module(global_feat, seen_att)  # [B, att_size]
        if mode == "test":
            return score

        if mode == "dot":
            score = self.base_module_dot(global_feat, seen_att)
            return score

        part_feat, atten_map, atten_attr, query = self.DSACA(feat)

        return score, global_feat, part_feat, atten_map, atten_attr, query
