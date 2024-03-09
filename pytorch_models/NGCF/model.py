import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class NGCFLayer(nn.Module):
    def __init__(self, in_size, out_size, norm_dict, dropout):
        super(NGCFLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        # weights for different types of messages
        self.W1 = nn.Linear(in_size, out_size, bias=True)
        self.W2 = nn.Linear(in_size, out_size, bias=True)

        # leaky relu
        self.leaky_relu = nn.LeakyReLU(0.2)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

        # initialization
        torch.nn.init.xavier_uniform_(self.W1.weight)
        torch.nn.init.constant_(self.W1.bias, 0)
        torch.nn.init.xavier_uniform_(self.W2.weight)
        torch.nn.init.constant_(self.W2.bias, 0)

        # norm
        self.norm_dict = norm_dict

    def forward(self, g, feat_dict):
        funcs = {}  # message and reduce functions dict
        # for each type of edges, compute messages and reduce them all
        for srctype, etype, dsttype in g.canonical_etypes:
            if srctype == dsttype:  # for self loops
                messages = self.W1(feat_dict[srctype])
                g.nodes[srctype].data[etype] = messages  # store in ndata
                funcs[(srctype, etype, dsttype)] = (
                    fn.copy_u(etype, "m"),
                    fn.sum("m", "h"),
                )  # define message and reduce functions
            else:
                src, dst = g.edges(etype=(srctype, etype, dsttype))
                norm = self.norm_dict[(srctype, etype, dsttype)]
                messages = norm * (
                    self.W1(feat_dict[srctype][src])
                    + self.W2(feat_dict[srctype][src] * feat_dict[dsttype][dst])
                )  # compute messages
                g.edges[(srctype, etype, dsttype)].data[
                    etype
                ] = messages  # store in edata
                funcs[(srctype, etype, dsttype)] = (
                    fn.copy_e(etype, "m"),
                    fn.sum("m", "h"),
                )  # define message and reduce functions

        g.multi_update_all(
            funcs, "sum"
        )  # update all, reduce by first type-wisely then across different types
        feature_dict = {}
        for ntype in g.ntypes:
            h = self.leaky_relu(g.nodes[ntype].data["h"])  # leaky relu
            h = self.dropout(h)  # dropout
            h = F.normalize(h, dim=1, p=2)  # l2 normalize
            feature_dict[ntype] = h
        return feature_dict


class NGCF(nn.Module):
    def __init__(self, g, in_size, layer_size, dropout, lmbd=1e-5):
        super(NGCF, self).__init__()
        self.lmbd = lmbd
        self.norm_dict = dict()
        for srctype, etype, dsttype in g.canonical_etypes:
            src, dst = g.edges(etype=(srctype, etype, dsttype))
            dst_degree = g.in_degrees(
                dst, etype=(srctype, etype, dsttype)
            ).float()  # obtain degrees
            src_degree = g.out_degrees(
                src, etype=(srctype, etype, dsttype)
            ).float()
            norm = torch.pow(src_degree * dst_degree, -0.5).unsqueeze(
                1
            )  # compute norm
            self.norm_dict[(srctype, etype, dsttype)] = norm

        self.layers = nn.ModuleList()
        self.layers.append(
            NGCFLayer(in_size, layer_size[0], self.norm_dict, dropout[0])
        )
        self.num_layers = len(layer_size)
        for i in range(self.num_layers - 1):
            self.layers.append(
                NGCFLayer(
                    layer_size[i],
                    layer_size[i + 1],
                    self.norm_dict,
                    dropout[i + 1],
                )
            )
        self.initializer = nn.init.xavier_uniform_

        # embeddings for different types of nodes
        self.feature_dict = nn.ParameterDict(
            {
                ntype: nn.Parameter(
                    self.initializer(torch.empty(g.num_nodes(ntype), in_size))
                )
                for ntype in g.ntypes
            }
        )

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = (users * pos_items).sum(1)
        neg_scores = (users * neg_items).sum(1)

        mf_loss = nn.LogSigmoid()(pos_scores - neg_scores).mean()
        mf_loss = -1 * mf_loss

        regularizer = (
            torch.norm(users) ** 2
            + torch.norm(pos_items) ** 2
            + torch.norm(neg_items) ** 2
        ) / 2
        emb_loss = self.lmbd * regularizer / users.shape[0]

        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, g, user_key, item_key, users, pos_items, neg_items):
        h_dict = {ntype: self.feature_dict[ntype] for ntype in g.ntypes}
        # obtain features of each layer and concatenate them all
        user_embeds = []
        item_embeds = []
        user_embeds.append(h_dict[user_key])
        item_embeds.append(h_dict[item_key])
        for layer in self.layers:
            h_dict = layer(g, h_dict)
            user_embeds.append(h_dict[user_key])
            item_embeds.append(h_dict[item_key])
        user_embd = torch.cat(user_embeds, 1)
        item_embd = torch.cat(item_embeds, 1)

        u_g_embeddings = user_embd[users, :]
        pos_i_g_embeddings = item_embd[pos_items, :]
        neg_i_g_embeddings = item_embd[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        
# Recommend new items by retraining interactions between new users and items
def bpr_loss(scores, neg_scores):
    return -torch.log(torch.sigmoid(scores - neg_scores)).mean()

def new_user_retrain_recommend(model, g, new_user_id, interacted_items, all_items, top_k=5, num_epochs=10, lr=0.0001):
    device = next(model.parameters()).device

    # Clone the model and graph to avoid modifying the original ones
    retrained_model = model.to(device)
    retrained_g = g.clone().to(device)

    # Add the new user to the graph
    retrained_g = dgl.add_nodes(retrained_g, 1, ntype='user')
    retrained_g.add_edges(new_user_id, interacted_items, etype='ui')

    # Create an optimizer for retraining
    optimizer = torch.optim.Adam(retrained_model.parameters(), lr=lr)

    # Training loop for retraining the model
    for epoch in range(num_epochs):
        retrained_model.train()  # Set the model in training mode
        optimizer.zero_grad()

        user_embeds, item_embeds, _ = retrained_model(retrained_g, 'user', 'item', retrained_g.nodes('user').data[dgl.NID], retrained_g.nodes('item').data[dgl.NID], None)
        # user_embeds, item_embeds, _ = retrained_model(retrained_g, "user", "item", new_user_id, interacted_items, None)
        
        # Predict the scores for the new user and all items
        scores = torch.matmul(user_embeds, item_embeds.t())

        # Exclude items that the new user has already interacted with
        for item_id in interacted_items:
            scores[0, item_id] = float('-inf')

        # Sample negative items (items not interacted by the new user)
        num_neg_samples = len(interacted_items)
        neg_items = random.sample([i for i in range(len(all_items)) if i not in interacted_items], num_neg_samples)

        # Calculate the BPR loss
        bpr_loss_value = bpr_loss(scores, scores[:, neg_items])

        bpr_loss_value.backward()
        optimizer.step()

    # Make predictions for the new user after retraining
    retrained_model.eval()  # Set the model in evaluation mode
    user_embeds, item_embeds, _ = retrained_model(retrained_g, 'user', 'item', retrained_g.nodes('user').data[dgl.NID], retrained_g.nodes('item').data[dgl.NID], None)

    # Get the indices of the top-k items
    top_items = torch.topk(item_embeds[0], k=top_k)[1].cpu().numpy()

    return top_items