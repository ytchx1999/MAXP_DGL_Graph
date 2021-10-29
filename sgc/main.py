import dgl
import dgl.function as fn
from utils import load_dgl_graph, time_diff
import torch
from tqdm import tqdm


def main():
    graph, labels, train_nid, val_nid, test_nid, node_feat = load_dgl_graph('../dataset')
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)

    sgc_list = []
    with graph.local_scope():
        # compute normalization
        degs = graph.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        norm = norm.to(node_feat.device).unsqueeze(1)
        # compute (D^-1 A^k D)^k X
        sgc_list.append(node_feat)
        for _ in tqdm(range(2)):
            node_feat = node_feat * norm
            graph.ndata['h'] = node_feat
            graph.update_all(fn.copy_u('h', 'm'),
                             fn.sum('m', 'h'))
            node_feat = graph.ndata.pop('h')
            node_feat = node_feat * norm

            sgc_list.append(node_feat)

    torch.save(sgc_list, '../dataset/sgc_emb.pt')


if __name__ == '__main__':
    main()
