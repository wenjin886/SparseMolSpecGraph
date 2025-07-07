import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch_geometric.nn import TransformerConv, global_mean_pool
import pytorch_lightning as pl

class NodeFeatureEncoder(nn.Module):
    def __init__(self, mult_class_num, nH_class_num, out_dim=32, hidden_dim=64):
        """
        args:
            mult_class_num: len(MULTIPLETS_LIST)
            nH_class_num: len(NUM_H)
            out_dim: 输出维度 of nodes
            hidden_dim: 隐藏层维度 of nodes
        """
        super().__init__()

        # Embedding for multiplicity (e.g., s, d, t, q, m) and nH
        self.mult_embed = nn.Embedding(mult_class_num, 4)
        self.nH_embed = nn.Embedding(nH_class_num, 4)


        # Learnable normalization layer: applies per-feature scale and bias
        self.feature_norm = nn.Linear(2 + 8, 2 + 8)  # input: 2 float + (4 mult_embed + 4 mult_embed)

        # Projection network to target embedding dim
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, centroid, width, nH, multiplet):
        """
        Inputs:
          - centroid, width, nH: [N, 1] float
          - multiplet: [N] long (class index)
        """
        # print('multiplet', type(multiplet), multiplet)
        mult_vec = self.mult_embed(multiplet)                 # [N, 4]
        nH_vec = self.nH_embed(nH)
       
        x = torch.cat([centroid.unsqueeze(-1), width.unsqueeze(-1), nH_vec,  mult_vec], dim=-1)  # [N, 2 + 4 + 4]
        # print('x', x.shape)

        x = self.feature_norm(x)  # Learnable normalization
        return self.fc(x)

class NMRGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, num_heads=2, edge_dim=1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 动态创建 TransformerConv 层
        self.conv_layers = nn.ModuleList()
        
        # 第一层：in_dim -> hidden_dim
        self.conv_layers.append(
            TransformerConv(in_dim, hidden_dim, heads=num_heads, edge_dim=edge_dim)
        )
        
        # 中间层：hidden_dim * 2 -> hidden_dim (因为 heads=2)
        for _ in range(1, num_layers):
            self.conv_layers.append(
                TransformerConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, edge_dim=edge_dim)
            )
        
        # 图级特征提取网络
        self.graph_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)  # 图级特征维度
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)  # 变成 [num_edges, 1]

        # 通过所有卷积层
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index, edge_attr)
            if i < self.num_layers - 1:  # 除了最后一层，都加激活函数
                x = F.relu(x)
        
        # 节点级特征
        node_embeddings = x
        
        # 图级特征：使用 global_mean_pool 聚合所有节点
        graph_embeddings = global_mean_pool(x, batch)
        graph_features = self.graph_encoder(graph_embeddings)
        
        return node_embeddings, graph_features

class MultiTaskNodePredictor(nn.Module):
    def __init__(self, node_dim, graph_dim, mult_class_num, nH_class_num):
        super().__init__()
        self.node_dim = node_dim
        self.graph_dim = graph_dim
        
        # 融合节点特征和图特征的网络
        self.fusion_network = nn.Sequential(
            nn.Linear(node_dim + graph_dim, node_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(node_dim, node_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 预测头
        self.fc_centroid = nn.Linear(node_dim // 2, 1)         # ppm
        self.fc_width = nn.Linear(node_dim // 2, 1)            # peak width
        self.fc_nH = nn.Linear(node_dim // 2, nH_class_num)               # proton count
        self.fc_mult = nn.Linear(node_dim // 2, mult_class_num)  # multiplicity (classification)

    def forward(self, masked_node_embeddings, graph_features, masked_node_batch):
        num_masked_nodes = masked_node_embeddings.size(0)
        
        # 根据batch信息为每个mask节点分配对应的图特征
        graph_features_per_masked_node = graph_features[masked_node_batch]
        
        # 融合节点特征和图特征
        combined_features = torch.cat([masked_node_embeddings, graph_features_per_masked_node], dim=-1)
        # print("masked_node_embeddings", masked_node_embeddings.shape, "graph_features_per_masked_node", graph_features_per_masked_node.shape)
        # print("combined_features", combined_features.shape)
        fused_features = self.fusion_network(combined_features)
        
        return {
            "centroid": self.fc_centroid(fused_features),
            "width": self.fc_width(fused_features),
            "nH": self.fc_nH(fused_features),
            "multiplet_logits": self.fc_mult(fused_features),
        }

class PeakGraphModule(pl.LightningModule):
    def __init__(self, mult_class_num, nH_class_num, 
                 in_dim=7, hidden_dim=32, num_layers=4, num_heads=2):
        """
        args:
            in_dim: 输入维度 of nodes
            hidden_dim: 隐藏层维度 of nodes
            num_layers: 编码器层数
            mult_class_num: len(MULTIPLETS_LIST)
        """
        super().__init__()
        self.save_hyperparameters()
        self.node_feature_encoder = NodeFeatureEncoder(mult_class_num, nH_class_num, in_dim)
        self.encoder = NMRGraphEncoder(in_dim, hidden_dim, num_layers, num_heads)
        # 修改预测器，传入图特征维度
        graph_dim = hidden_dim // 4  # 与 NMRGraphEncoder 中的图特征维度一致
        self.predictor = MultiTaskNodePredictor(hidden_dim*num_heads, graph_dim, mult_class_num, nH_class_num)

    def forward(self, data):
        node_features = self.node_feature_encoder(data.centroids, data.peak_widths, data.nH, data.multiplets)
        node_embeddings, graph_features = self.encoder(node_features, data.edge_index, data.batch, data.edge_attr)
        
        # 传入节点特征和图特征
        masked_node_batch = data.batch[data.masked_node_index]
        # print("masked_node_batch", masked_node_batch)
        pred = self.predictor(node_embeddings[data.masked_node_index], graph_features, masked_node_batch)
        return pred
    
    def compute_multitask_loss(self, pred, target, weight_dict=None):
        loss_centroid = F.mse_loss(pred["centroid"].squeeze(), target["centroid"])
        loss_width = F.mse_loss(pred["width"].squeeze(), target["width"])
        # print(target["nH"])
        # print(target["multiplet"])
        loss_nH = F.cross_entropy(pred["nH"], target["nH"])
        
        loss_mult = F.cross_entropy(pred["multiplet_logits"], target["multiplet"])

        if weight_dict is None:
            weight_dict = {"centroid": 1.0, "width": 1.0, "nH": 1.0, "multiplet": 1.0}

        total_loss = (
            weight_dict["centroid"] * loss_centroid +
            weight_dict["width"] * loss_width +
            weight_dict["nH"] * loss_nH +
            weight_dict["multiplet"] * loss_mult
        )
        return total_loss, {"loss_centroid": loss_centroid, "loss_width": loss_width, "loss_nH": loss_nH, "loss_mult": loss_mult}
    
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
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)