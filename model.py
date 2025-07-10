import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch_geometric.nn import TransformerConv, global_mean_pool
from torchmetrics.classification import MulticlassAUROC
import pytorch_lightning as pl
import copy


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class NodeFeatureEmbedding(nn.Module):
    def __init__(self, mult_class_num, nH_class_num, 
                 c_w_embed_dim=8, mult_embed_dim=32, nH_embed_dim=16):
                #  out_dim=32, hidden_dim=64):
        """
        args:
            mult_class_num: len(MULTIPLETS_LIST)
            nH_class_num: len(NUM_H)
            c_w_embed_dim: 中心和宽度 embedding 维度
            mult_embed_dim: 多重性 embedding 维度
            nH_embed_dim: 氢原子数 embedding 维度
            out_dim: 输出维度 of nodes
            hidden_dim: 隐藏层维度 of nodes
        """
        super().__init__()

        # Embedding for multiplicity (e.g., s, d, t, q, m) and nH
        self.mult_embed = nn.Embedding(mult_class_num, mult_embed_dim)
        self.nH_embed = nn.Embedding(nH_class_num, nH_embed_dim)
        continuous_value_embed = nn.Sequential(
            nn.Linear(1, 1), # norm
            nn.SiLU(),
            nn.Linear(1, c_w_embed_dim)
        )
        c = copy.deepcopy
        self.centroid_embed = c(continuous_value_embed)
        self.width_embed = c(continuous_value_embed)
        node_feature_dim = mult_embed_dim + nH_embed_dim + c_w_embed_dim*2
        self.linear = nn.Linear(node_feature_dim, node_feature_dim)


       
    def forward(self, centroid, width, nH, multiplet):
        """
        Inputs:
          - centroid, width, nH: [N, 1] float
          - multiplet: [N] long (class index)
        """
        mult_vec = self.mult_embed(multiplet)               
        nH_vec = self.nH_embed(nH)
        cen_vec = self.centroid_embed(centroid.unsqueeze(-1))
        wid_vec = self.width_embed(width.unsqueeze(-1))
        x = torch.cat([cen_vec, wid_vec, nH_vec,  mult_vec], dim=-1)  # [N,all_feature_dim]
        return self.linear(x).relu()
      

class NMRGraphEncoder(nn.Module):
    def __init__(self, in_node_dim, hidden_node_dim, 
                #  graph_dim,
                 num_layers, num_heads, edge_dim=1):
        super().__init__()
        self.in_node_dim = in_node_dim
        self.hidden_node_dim = hidden_node_dim
        self.num_layers = num_layers
        
        # self.edge_embed = nn.Linear(1, 1) # norm
        self.conv_layers = clones(TransformerConv(in_node_dim, hidden_node_dim, heads=num_heads, edge_dim=edge_dim), num_layers)
        self.sublayers = clones(SublayerConnection(size=in_node_dim, dropout=0.1), num_layers)
        
        self.linear = nn.Linear(in_node_dim, in_node_dim)
        

    def forward(self, x, edge_index, batch, edge_attr):
        
        edge_attr = edge_attr.unsqueeze(-1)  # 变成 [num_edges, 1]

        # 通过所有卷积层
        for i, conv in enumerate(self.conv_layers):
            x = self.sublayers[i](x, lambda x: conv(x, edge_index, edge_attr))
        
        # 节点级特征
        node_embeddings = self.linear(x)
        
        return node_embeddings

class MultiTaskNodePredictor(nn.Module):
    def __init__(self, node_dim, mult_class_num, nH_class_num,
                mult_embed_dim, nH_embed_dim, cen_embed_dim, wid_embed_dim):
        super().__init__()
        self.node_dim = node_dim
        
       
        # 预测头
        self.fc_centroid = nn.Sequential(nn.Linear(node_dim, node_dim), nn.SiLU(), 
                                         nn.Linear(node_dim, cen_embed_dim), nn.SiLU(), 
                                         nn.Linear(cen_embed_dim, 1))         # ppm
        self.fc_width = nn.Sequential(nn.Linear(node_dim, node_dim), nn.SiLU(), 
                                      nn.Linear(node_dim, wid_embed_dim), nn.SiLU(), 
                                      nn.Linear(wid_embed_dim, 1))            # peak width
        self.fc_nH = nn.Sequential(nn.Linear(node_dim, node_dim), nn.SiLU(), 
                                   nn.Linear(node_dim, nH_embed_dim), nn.SiLU(), 
                                   nn.Linear(nH_embed_dim, nH_class_num))               # proton count
        self.fc_mult = nn.Sequential(nn.Linear(node_dim, node_dim), nn.SiLU(), 
                                     nn.Linear(node_dim, mult_embed_dim), nn.SiLU(), 
                                     nn.Linear(mult_embed_dim, mult_class_num))  # multiplicity (classification)


    def forward(self, masked_node_embeddings):

        return {
            "centroid": self.fc_centroid(masked_node_embeddings),
            "width": self.fc_width(masked_node_embeddings),
            "nH": self.fc_nH(masked_node_embeddings),
            "multiplet_logits": self.fc_mult(masked_node_embeddings),
        }

class PeakGraphModule(pl.LightningModule):
    def __init__(self, mult_class_num, nH_class_num, 
                 mult_embed_dim=16, nH_embed_dim=8, c_w_embed_dim=8,
                 num_layers=4, num_heads=4,
                 mult_class_weights=None,
                 warm_up_step=None, lr=None):
        """
        args:
            mult_class_num: len(MULTIPLETS_LIST)
            nH_class_num: len(NUM_H_LIST)
            in_node_dim: graph encoder 中 node feature 的输入维度
            hidden_dim: graph encoder 中 node feature 的隐藏层维度
            num_layers: graph encoder 中 编码器层数
            num_heads: graph encoder 中 多头注意力机制的 head 数
        """
        super().__init__()
        self.save_hyperparameters()
        self.node_feature_encoder = NodeFeatureEmbedding(mult_class_num, nH_class_num, 
                                                       mult_embed_dim=mult_embed_dim, 
                                                       nH_embed_dim=nH_embed_dim,
                                                       c_w_embed_dim=c_w_embed_dim,
                                                       )
        in_node_dim = c_w_embed_dim*2 + mult_embed_dim + nH_embed_dim
        print('in_node_dim of node feature encoder', in_node_dim)
        assert in_node_dim % num_heads == 0, "in_node_dim must be divisible by num_heads"
        hidden_node_dim = in_node_dim // num_heads
        self.encoder = NMRGraphEncoder(in_node_dim, hidden_node_dim, num_layers, num_heads)
        self.predictor = MultiTaskNodePredictor(in_node_dim, mult_class_num, nH_class_num,
                                                mult_embed_dim, nH_embed_dim, c_w_embed_dim, c_w_embed_dim)
        self.in_node_dim = in_node_dim

        self.warm_up_step = warm_up_step
        self.lr = lr
        self.mult_class_weights = mult_class_weights
        self.__init_weights__()

        self.mult_auroc = MulticlassAUROC(mult_class_num, average="macro")
        self.nH_auroc = MulticlassAUROC(nH_class_num, average="macro")
    
    def __init_weights__(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    
    def encode(self, data):
        node_features = self.node_feature_encoder(data.centroids, data.peak_widths, data.nH, data.multiplets)
        node_embeddings = self.encoder(node_features, data.edge_index, data.batch, data.edge_attr)
        return node_embeddings

    def forward(self, data):
        node_embeddings = self.encode(data)
        
        # 传入节点特征和图特征
        pred = self.predictor(node_embeddings[data.masked_node_index])
        return pred
    
    def compute_multitask_loss(self, pred, target, weight_dict=None):
        loss_centroid = F.mse_loss(pred["centroid"].squeeze(), target["centroid"])
        loss_width = F.mse_loss(pred["width"].squeeze(), target["width"])
        loss_nH = F.cross_entropy(pred["nH"], target["nH"])

        if self.mult_class_weights is not None:
            self.mult_class_weights.to(target["multiplet"].device)

        loss_mult = F.cross_entropy(pred["multiplet_logits"], target["multiplet"],
                                    weight=self.mult_class_weights)

        if weight_dict is None:
            weight_dict = {"centroid": 1.0, "width": 1.0, "nH": 1.0, "multiplet": 1.0}

        total_loss = (
            weight_dict["centroid"] * loss_centroid +
            weight_dict["width"] * loss_width +
            weight_dict["nH"] * loss_nH +
            weight_dict["multiplet"] * loss_mult
        )
        return total_loss, {"loss_centroid": loss_centroid, "loss_width": loss_width, "loss_nH": loss_nH, "loss_mult": loss_mult}
    
    def compute_accuracy_auc(self, pred, target):
        pred_nH = torch.argmax(pred["nH"], dim=-1)
        acc_nH = (pred_nH == target["nH"]).float().mean()
        pred_mult = torch.argmax(pred["multiplet_logits"], dim=-1)
        acc_mult = (pred_mult == target["multiplet"]).float().mean()

        nH_probs = torch.softmax(pred["nH"], dim=-1)
        nH_auc = self.nH_auroc(nH_probs, target["nH"])
        mult_probs = torch.softmax(pred["multiplet_logits"], dim=-1)
        mult_auc = self.mult_auroc(mult_probs, target["multiplet"])
        return acc_nH, acc_mult, nH_auc, mult_auc
    
    def training_step(self, batch, batch_idx):
        
        batch_size = len(batch.id)
        pred = self(batch)

        # 提取目标：batch.masked_node_target 是 dict of tensors
        loss, loss_dic = self.compute_multitask_loss(pred, batch.masked_node_target)
        self.log("train_loss", loss, batch_size=batch_size)
        self.log("train_loss_centroid", loss_dic["loss_centroid"], batch_size=batch_size)
        self.log("train_loss_width", loss_dic["loss_width"], batch_size=batch_size)
        self.log("train_loss_nH", loss_dic["loss_nH"], batch_size=batch_size)
        self.log("train_loss_mult", loss_dic["loss_mult"], batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_size = len(batch.id)
        pred = self(batch)

        # 提取目标：batch.masked_node_target 是 dict of tensors
        loss, loss_dic = self.compute_multitask_loss(pred, batch.masked_node_target)
        self.log("val_loss", loss, batch_size=batch_size)
        self.log("val_loss_centroid", loss_dic["loss_centroid"], batch_size=batch_size)
        self.log("val_loss_width", loss_dic["loss_width"], batch_size=batch_size)
        self.log("val_loss_nH", loss_dic["loss_nH"], batch_size=batch_size)
        self.log("val_loss_mult", loss_dic["loss_mult"], batch_size=batch_size)
        acc_nH, acc_mult, nH_auc, mult_auc = self.compute_accuracy_auc(pred, batch.masked_node_target)
        self.log("val_acc_nH", acc_nH, batch_size=batch_size)
        self.log("val_auc_nH", nH_auc, batch_size=batch_size)
        self.log("val_acc_mult", acc_mult, batch_size=batch_size)
        self.log("val_auc_mult", mult_auc, batch_size=batch_size)
        return loss
    
    def test_step(self, batch, batch_idx):
        batch_size = len(batch.id)
        pred = self(batch)

        # 提取目标：batch.masked_node_target 是 dict of tensors
        loss, loss_dic = self.compute_multitask_loss(pred, batch.masked_node_target)
        self.log("test_loss", loss, batch_size=batch_size)
        self.log("test_loss_centroid", loss_dic["loss_centroid"], batch_size=batch_size)
        self.log("test_loss_width", loss_dic["loss_width"], batch_size=batch_size)
        self.log("test_loss_nH", loss_dic["loss_nH"], batch_size=batch_size)
        self.log("test_loss_mult", loss_dic["loss_mult"], batch_size=batch_size)
        acc_nH, acc_mult, nH_auc, mult_auc = self.compute_accuracy_auc(pred, batch.masked_node_target)
        self.log("test_acc_nH", acc_nH, batch_size=batch_size)
        self.log("test_auc_nH", nH_auc, batch_size=batch_size)
        self.log("test_acc_mult", acc_mult, batch_size=batch_size)
        self.log("test_auc_mult", mult_auc, batch_size=batch_size)
        return loss

    
    
    def configure_optimizers(self):
        # 使用 AdamW 优化器
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            amsgrad=True,
            weight_decay=1e-12
        )

        def rate(step):
           
            if step == 0:
                step = 1
            lr_scale = 1 * (
                self.in_node_dim ** (-0.5) * min(step ** (-0.5), step * self.warm_up_step ** (-1.5))
            )

            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=rate
        )

        # 返回优化器和调度器
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    