import torch
import torch.nn as nn


class cpt_loss(nn.Module):
    def __init__(self, device) -> None:
        super(cpt_loss, self).__init__()

        self.device = device

    def forward(self, atten_map):
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


def get_attr_group(name):
    if "CUB" in name:
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

    elif "AwA" in name:
        attr_group = {
            1: [i for i in range(0, 8)],
            2: [i for i in range(8, 14)],
            3: [i for i in range(14, 18)],
            4: [i for i in range(18, 34)]+[44, 45],
            5: [i for i in range(34, 44)],
            6: [i for i in range(46, 51)],
            7: [i for i in range(51, 63)],
            8: [i for i in range(63, 78)],
            9: [i for i in range(78, 85)],
        }

    elif "SUN" in name:
        attr_group = {
            1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
            2: [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73]+[80, 99],
            3: [74, 75, 76, 77, 78, 79, 81, 82, 83, 84, ],
            4: [85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98] + [100, 101]
        }
    else:
        attr_group = {}

    return attr_group


class ad_loss(nn.Module):
    def __init__(self, dataset_name) -> None:
        super(ad_loss, self).__init__()

        self.attr_group = get_attr_group(dataset_name)

    def forward(self, query):
        loss_sum = 0
        for key in self.attr_group:
            group = self.attr_group[key]
            proto_each_group = query[group]  # g1 * v
            channel_l2_norm = torch.norm(proto_each_group, p=2, dim=0)
            loss_sum += channel_l2_norm.mean()

        loss_sum = loss_sum.float()/len(self.attr_group)

        return loss_sum
