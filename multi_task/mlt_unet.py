import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class DWConv(nn.Module):
    def __init__(self, dim=768,group_num=4):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim//group_num)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def Conv1X1(in_, out):
    return torch.nn.Conv2d(in_, out, 1, padding=0)


def Conv3X3(in_, out):
    return torch.nn.Conv2d(in_, out, 3, padding=1)


class Mlp(nn.Module):
    def __init__(self, in_features, out_features, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = out_features // 4
        self.fc1 = Conv1X1(in_features, hidden_features)
        self.gn1=nn.GroupNorm(hidden_features//4,hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.gn2 = nn.GroupNorm(hidden_features // 4, hidden_features)
        self.act = act_layer()
        self.fc2 = Conv1X1(hidden_features, out_features)
        self.gn3=nn.GroupNorm(out_features//4,out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x=self.gn1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x=self.gn2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x=self.gn3(x)
        x = self.drop(x)
        return x


class LocalSABlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, k=16, u=1, m=7):
        super(LocalSABlock, self).__init__()
        self.kk, self.uu, self.vv, self.mm, self.heads = k, u, out_channels // heads, m, heads
        self.padding = (m - 1) // 2

        self.queries = nn.Sequential(
            nn.Conv2d(in_channels, k * heads, kernel_size=1, bias=False),
            nn.GroupNorm(k*heads//4,k*heads)
        )
        self.keys = nn.Sequential(
            nn.Conv2d(in_channels, k * u, kernel_size=1, bias=False),
            nn.GroupNorm(k*u//4,k*u)
        )
        self.values = nn.Sequential(
            nn.Conv2d(in_channels, self.vv * u, kernel_size=1, bias=False),
            nn.GroupNorm(self.vv*u//4,self.vv*u)
        )

        self.softmax = nn.Softmax(dim=-1)

        self.embedding = nn.Parameter(torch.randn([self.kk, self.uu, 1, m, m]), requires_grad=True)

    def forward(self, x):
        n_batch, C, w, h = x.size()
        queries = self.queries(x).view(n_batch, self.heads, self.kk, w * h)  # b, heads, k , w * h
        softmax = self.softmax(self.keys(x).view(n_batch, self.kk, self.uu, w * h))  # b, k, uu, w * h
        values = self.values(x).view(n_batch, self.vv, self.uu, w * h)  # b, v, uu, w * h
        content = torch.einsum('bkum,bvum->bkv', (softmax, values))
        content = torch.einsum('bhkn,bkv->bhvn', (queries, content))
        values = values.view(n_batch, self.uu, -1, w, h)
        context = F.conv3d(values, self.embedding, padding=(0, self.padding, self.padding))
        context = context.view(n_batch, self.kk, self.vv, w * h)
        context = torch.einsum('bhkn,bkvn->bhvn', (queries, context))

        out = content + context
        out = out.contiguous().view(n_batch, -1, w, h)

        return out


class TFBlock(nn.Module):

    def __init__(self, in_chnnels, out_chnnels, mlp_ratio=2., drop=0.3,
                 drop_path=0., act_layer=nn.GELU, linear=False):
        super(TFBlock, self).__init__()
        self.in_chnnels = in_chnnels
        self.out_chnnels = out_chnnels
        self.attn = LocalSABlock(
            in_channels=in_chnnels, out_channels=out_chnnels
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=in_chnnels, out_features=out_chnnels, act_layer=act_layer, drop=drop, linear=linear)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        hidden_planes = max(planes,in_planes) // self.expansion
        self.conv1 = nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(hidden_planes //4,
                                hidden_planes)
        self.conv2 = nn.ModuleList([TFBlock(hidden_planes, hidden_planes)])
        self.bn2 = nn.GroupNorm(hidden_planes // 4,
                                hidden_planes)
        self.conv2.append(nn.GELU())
        self.conv2 = nn.Sequential(*self.conv2)
        self.conv3 = nn.Conv2d(hidden_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(planes // 4, planes)
        self.GELU=nn.GELU()
        self.shortcut = nn.Sequential()
        if in_planes!=planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                nn.GroupNorm(planes//4,planes)
            )
    def forward(self, x):
        out = self.GELU(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.GELU(self.bn3(self.conv3(out)))
        out += self.shortcut(x)
        return out


class Trans_EB(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Bottleneck(in_, out)
        self.activation=torch.nn.GELU()
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Conv3X3(in_, out)
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class LABlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(LABlock, self).__init__()
        self.W_1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(output_channels//4,output_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(output_channels//4,output_channels),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.gelu=nn.GELU()
    def forward(self, inputs):
        sum = 0
        for input in inputs:
            sum += input
        sum=self.gelu(sum)
        out = self.W_1(sum)
        psi = self.psi(out)  # Mask
        return psi

class Down1(nn.Module):

    def __init__(self):
        super(Down1, self).__init__()
        self.nn1 = ConvRelu(64, 64)
        self.nn2 = Trans_EB(64, 64)
        self.patch_embed = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, inputs):
        scale1_1 = self.nn1(inputs)
        scale1_2 = self.nn2(scale1_1)
        unpooled_shape = scale1_2.size()
        outputs, indices = self.patch_embed(scale1_2)
        # return outputs, indices, unpooled_shape, scale1_1, scale1_2
        return  scale1_2


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        h = self.proj(h)

        return x + h


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channel=64, inc_channel=32, beta=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, inc_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channel + inc_channel, inc_channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channel + 2 * inc_channel, inc_channel, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channel + 3 * inc_channel, inc_channel, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channel + 4 * inc_channel, in_channel, 3, 1, 1)
        self.lrelu = nn.LeakyReLU()
        self.Down1 = Down1()#dim, n_cls

        self.b = beta

    def forward(self, x):
        # print("x.shape") #torch.Size([5, 64, 256, 256])
        #
        # print(x.shape)
        block1 = self.lrelu(self.conv1(x))
        block2 = self.lrelu(self.conv2(torch.cat((block1, x), dim=1)))
        block3 = self.lrelu(self.conv3(torch.cat((block2, block1, x), dim=1)))
        block4 = self.lrelu(self.conv4(torch.cat((block3, block2, block1, x), dim=1)))
        trans1 = self.Down1(x)
        # print("trans1.shape")
        # print(trans1.shape)
        x = x + trans1
        # out = self.conv5(torch.cat((block4, block3, block2, block1, x), dim=1))
        out = self.conv5(torch.cat((block4, block3, block2, block1, x), dim=1))
        # print("out.shape")
        # print(out.shape)

        return x + self.b * out


class RRDB(nn.Module):
    def __init__(self, in_channel=64, inc_channel=32, beta=0.2):
        super().__init__()
        self.RDB = ResidualDenseBlock(in_channel, inc_channel)
        self.b = beta

    def forward(self, x):
        out = self.RDB(x)
        # out = self.RDB(out)
        # out = self.RDB(out)

        return x + self.b * out


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.x_head = nn.Conv2d(1, ch, kernel_size=3, stride=1, padding=1)
        self.seg_head = nn.Sequential(
            nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1),
            RRDB(in_channel=ch)
        )

        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.noisy_upblocks = nn.ModuleList()
        self.seg_upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                cat_ch = chs.pop()
                self.noisy_upblocks.append(ResBlock(
                    in_ch=cat_ch + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                self.seg_upblocks.append(ResBlock(
                    in_ch=cat_ch + now_ch + out_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))

                now_ch = out_ch
            if i != 0:
                self.noisy_upblocks.append(UpSample(now_ch))
                self.seg_upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.noisy_tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 1, 3, stride=1, padding=1)
        )
        self.seg_tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 1, 3, stride=1, padding=1)
            # nn.Sigmoid()
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.x_head.weight)
        init.zeros_(self.x_head.bias)

        init.xavier_uniform_(self.noisy_tail[-1].weight, gain=1e-5)
        init.zeros_(self.noisy_tail[-1].bias)
        init.xavier_uniform_(self.seg_tail[-1].weight, gain=1e-5)
        init.zeros_(self.seg_tail[-1].bias)

    def forward(self, x_t, t, image, entropy=None):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        # print(self.x_head(x).shape)
        # print(self.image_head(image).shape)
        # print(image.shape)
        # print(entropy.shape)
        if entropy != None:
            entropy= self.seg_head(entropy)
            img_seg = self.seg_head(image)
            img_ent =   entropy + img_seg
            h = self.x_head(x_t) + img_ent
        else:
            h = self.x_head(x_t) + self.seg_head(image)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        bottom = h
        # Upsampling
        skip_count = len(hs)
        h_seg = []
        for layer in self.noisy_upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs[skip_count - 1]], dim=1)
                skip_count -= 1
                h = layer(h, temb)
                h_seg.append(h)
            else:
                h = layer(h, temb)
        assert skip_count == 0

        for layer in self.seg_upblocks:
            if isinstance(layer, ResBlock):
                bottom = torch.cat([bottom, hs.pop(), h_seg[skip_count]], dim=1)
                skip_count += 1
            bottom = layer(bottom, temb)
        h = self.noisy_tail(h)
        bottom = self.seg_tail(bottom)
        assert len(hs) == 0

        return h, bottom


if __name__ == '__main__':
    batch_size = 1
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1).to("cuda:0")
    x = torch.randn(batch_size, 1, 32, 32).to("cuda:0")
    image = torch.randn(batch_size, 4, 32, 32).to("cuda:0")
    t = torch.randint(1000, (batch_size, )).to("cuda:0")
    y1, y2 = model(x, t, image)
    print(y1.shape, y2.shape)

