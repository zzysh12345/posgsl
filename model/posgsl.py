import time
from copy import deepcopy
import os
import sys
from torch_geometric.data import Dataset as PygDataset, Data, Batch
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from opengsl.module.encoder import MLPEncoder, GNNEncoder
from opengsl.module.solver import Solver
from opengsl.utils.recorder import Recorder
# from opengsl.module.model.gmt import GMT
# from opengsl.module.model.topkpool import TopKPool
# from opengsl.module.model.edgepool import EdgePool
from .parsing import SimpleParsingModule
from .selecting import SubgraphSelectModule, ProjectionEstimator
from .gsl import BasicSubGraphLearner, MotifVector, BasicGraphLearner, PadSubGraphLearner
# from .label import LabelInformedGraphLearner
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.utils import to_dense_adj, subgraph, cumsum, scatter
import numpy as np


class POSGSL(nn.Module):
    def __init__(self, conf, n_feat, n_class):
        super(POSGSL, self).__init__()
        self.conf = conf
        self.n_feat = n_feat
        self.n_class = n_class
        # self.use_select = self.conf.use_select and self.conf.parsing_module['parsing'] != 'original'
        self.use_gsl = self.conf.use_gsl
        self.use_motif = self.conf.use_motif
        self.use_seperate_encoder = self.conf.use_seperate_encoder and self.use_motif
        self.version = self.conf.version   # 使用原来的全batch一起生成结构，or使用padding的方式
        self.recover_full_adj = self.conf.recover_full_adj   # 是否恢复原图，or只基于学习的子图
        self.use_gate = self.conf.use_gate and self.conf.parsing_module['parsing'] != 'original'
        self.use_residual = 'use_residual' in self.conf and self.conf.use_residual

        # parse
        self.parsing = SimpleParsingModule(**conf.parsing_module, return_new_mapping=(conf.use_gsl and conf.version==0),
                                           require_full_edge=(conf.use_gsl and conf.version==0))
        # # select
        # self.selecting = SubgraphSelectModule(n_feat=self.n_feat, select_flag=self.use_select, **conf.selecting_module)
        # gsl
        if self.version == 0:
            self.gsl = BasicSubGraphLearner(n_feat=self.n_feat, n_hidden=self.conf.backbone['n_hidden'], **conf.gsl_module)
        elif self.version == 1:
            self.gsl = PadSubGraphLearner(n_feat=self.n_feat, n_hidden=self.conf.backbone['n_hidden'], **conf.gsl_module)
        # classifier
        self.backbone = GNNEncoder(n_feat=self.n_feat, n_class=self.n_class, **conf.backbone)
        # self.classifier = MLPEncoder(n_feat=self.selecting.conf_encoder['n_class'], n_class=n_class, **conf.classifier)
        self.backbone2 = GNNEncoder(n_feat=self.n_feat, n_class=self.n_class, **conf.backbone)
        # gate
        if self.use_gate:
            self.gate = ProjectionEstimator(n_feat=self.backbone2.output_linear.n_feat, act='softmax')
        # motif
        device = torch.device('cuda') if not ('use_cpu' in conf and conf.use_cpu) else torch.device('cpu')
        self.motif = MotifVector(n_hidden=self.backbone.output_linear.n_feat, n_class=n_class, device=device, **conf.motif)

        self.forward = self.forward if self.use_gsl else self.forward_wo_gsl

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

    def forward_wo_gsl(self, batch, return_sub=False, only_backbone=False):
        if only_backbone:
            return self.backbone(x=batch.x, edge_index=batch.edge_index, batch=batch.batch), 0
        else:
            t0 = time.time()
            batch_subs, belong, new_mapping = self.parsing(batch)
            # print(time.time()-t0)
            # ********** select 相关 **********
            # selected_x, selected_z_node, selected_z_sub, selected_edge_index, selected_batch, selected_belong, selected_mapping, selected_score, _ = self.selecting(batch_subs, belong, new_mapping)
            # ********** select 相关 **********
            encoder = self.backbone2 if self.use_seperate_encoder else self.backbone
            z_sub = encoder(batch_subs.x, edge_index=batch_subs.edge_index, batch=batch_subs.batch, get_cls=False)
            # z_sub_agg = z_sub * selected_score.unsqueeze(1).expand(-1, z_sub.shape[1])
            score = 0
            if self.use_gate:
                score = self.gate(z_sub, belong)
                z_sub = z_sub * score.unsqueeze(1)
                z_global = scatter(z_sub, belong, reduce='sum')
            else:
                z_global = scatter(z_sub, belong, reduce='mean')
            output = encoder.output_linear(z_global)
            # print(time.time() - t0)
            if return_sub:
                return output, 0, z_sub, belong, score
            else:
                return output, 0

    def forward(self, batch, return_sub=False):
        score = 0
        batch_subs, belong, new_mapping = self.parsing(batch)
        # ********** select 相关 **********
        # selected_x, selected_z_node, selected_z_sub, selected_edge_index, selected_batch, selected_belong, selected_mapping, selected_score, full_edge_index = self.selecting(batch_subs, belong, new_mapping)
        # x_gsl = selected_x if self.gsl.encoder_type else selected_z_node
        # ********** select 相关 **********

        if self.recover_full_adj:
            edge_index_out_subs, edge_attr_out_subs, edge_index_out, edge_attr_out = \
                self.gsl(batch_subs.x, selected_edge_index=batch_subs.edge_index, selected_batch=batch_subs.batch,
                         selected_mapping=new_mapping, selected_belong=belong, selected_score=None,
                         full_edge_index=batch_subs.full_edge_index if hasattr(batch_subs, 'full_edge_index') else None,
                         recover_full_adj=True, raw_edge_index=batch.edge_index)
            output = self.backbone(batch.x, edge_index=edge_index_out, edge_attr=edge_attr_out, batch=batch.batch)
        else:
            edge_index_out_subs, edge_attr_out_subs = \
                self.gsl(batch_subs.x, selected_edge_index=batch_subs.edge_index, selected_batch=batch_subs.batch,
                         selected_mapping=new_mapping, selected_belong=belong, selected_score=None,
                         full_edge_index=batch_subs.full_edge_index if hasattr(batch_subs, 'full_edge_index') else None,
                         recover_full_adj=False, raw_edge_index=batch.edge_index)
            z_sub = self.backbone(batch_subs.x, edge_index=edge_index_out_subs, edge_attr=edge_attr_out_subs,
                                   batch=batch_subs.batch, get_cls=False)
            if self.use_gate:
                score = self.gate(z_sub, belong)
                output = z_sub * score.unsqueeze(1)
                output = scatter(output, belong, reduce='sum')
            else:
                output = scatter(z_sub, belong, reduce='mean')
            if self.use_residual:
                lamb = self.conf.gsl_module['lamb']
                output = lamb * output + (1-lamb) * self.backbone(batch.x, edge_index=batch.edge_index, batch=batch.batch, get_cls=False)
            output = self.backbone.output_linear(output)

        loss_con = 0
        if self.use_motif:
            if self.use_seperate_encoder:
                z_sub = self.backbone2(batch_subs.x, edge_index=edge_index_out_subs, edge_attr=edge_attr_out_subs,
                                      batch=batch_subs.batch, get_cls=False)
            else:
                z_sub = z_sub
            y_sub = batch.y[belong]
            loss_con = self.motif.loss_contrastive(z_sub, y_sub)
        if return_sub:
            return output, loss_con, z_sub, belong, score
        else:
            return output, loss_con

class POSGSLSolver(Solver):
    def __init__(self, conf, dataset):
        super(POSGSLSolver, self).__init__(conf, dataset)
        self.method_name = 'modgsl'
        self.model = POSGSL(self.conf, self.dim_feats, self.n_classes).to(self.device)
        self.model.parsing.init_parsing(self.dataset)
        self.pretrain_flag = 'n_pretrain' in self.conf.training and self.conf.training['n_pretrain'] > 0
        if self.pretrain_flag:
            assert self.conf.use_gsl

    def learn_gc(self, debug=False):
        if isinstance(self.dataset.data_raw, PygDataset):
            train_dataset = self.dataset.data_raw[self.train_mask]
            test_dataset = self.dataset.data_raw[self.test_mask]
            val_dataset = self.dataset.data_raw[self.val_mask]
        elif isinstance(self.dataset.data_raw, list):
            train_dataset = [self.dataset.data_raw[idx] for idx in self.train_mask.tolist()]
            test_dataset = [self.dataset.data_raw[idx] for idx in self.test_mask.tolist()]
            val_dataset = [self.dataset.data_raw[idx] for idx in self.val_mask.tolist()]
        else:
            raise NotImplementedError

        if self.pretrain_flag:
            self.model.backbone.requires_grad_(True)
            self.pretrain(debug=debug)
            torch.cuda.empty_cache()
            self.recoder = Recorder(self.conf.training['patience'], self.conf.training['criterion'])
            if 'mode' in self.conf and self.conf.mode == 'testtime':
                self.model.backbone.requires_grad_(False)
                self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'], weight_decay=self.conf.training['weight_decay'])
        if self.conf.use_motif:
            if 'init_motif' not in self.conf.motif or self.conf.motif['init_motif']:
                print('init motif from pretrained')
                self.init_motif(train_dataset)

        train_loader = DataLoader(train_dataset, self.conf.training['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, self.conf.training['test_batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, self.conf.training['test_batch_size'], shuffle=False)
        time_train_list = []

        for epoch in range(self.conf.training['n_epochs']):

            if self.conf.use_motif and epoch >= self.conf.training['n_motif_update_min'] and epoch % self.conf.training['n_motif_update_each'] == 0:
                if 'update_motif' not in self.conf.motif or self.conf.motif['update_motif']:
                    if debug:
                        print('Motif Projection')
                    self.update_motif(train_dataset)

            improve = ''
            t0 = time.time()
            loss_train = 0

            # forward and backward
            preds = []
            ground_truth = []
            self.model.train()
            for batch in train_loader:
                # print(batch)
                # print(torch.cuda.memory_allocated()/1048576)
                self.optim.zero_grad()
                batch = batch.to(self.device)
                out, loss_con = self.model(batch)
                loss = self.loss_fn(out, batch.y.view(-1))
                # print(torch.cuda.memory_allocated() / 1048576)
                if self.conf.use_motif:
                    loss += loss_con * self.conf.training['lambda']
                loss.backward()
                self.optim.step()
                loss_train += loss.item() * batch.num_graphs
                pred = F.softmax(out, dim=1)
                if self.conf.training['metric'] != 'acc':
                    pred = pred[:,1].unsqueeze(1)
                preds.append(pred.detach().cpu())
                ground_truth.append(batch.y.detach().cpu().unsqueeze(1))
                # print(time.time()-t1)
            preds = torch.vstack(preds).squeeze().numpy()
            ground_truth = torch.vstack(ground_truth).squeeze().numpy()
            loss_train = loss_train / len(train_loader.dataset)
            acc_train = self.metric(ground_truth, preds)
            time_train = time.time() - t0
            time_train_list.append(time_train)

            # Evaluate
            preds = []
            ground_truth = []
            self.model.eval()
            loss_val = 0
            for batch in val_loader:
                batch = batch.to(self.device)
                with torch.no_grad():
                    out, _ = self.model(batch)
                    pred = F.softmax(out, dim=1)
                    if self.conf.training['metric'] != 'acc':
                        pred = pred[:, 1].unsqueeze(1)
                    preds.append(pred.detach().cpu())
                    ground_truth.append(batch.y.detach().cpu().unsqueeze(1))
                loss_val += self.loss_fn(out, batch.y.view(-1), reduction='sum').item()
            preds = torch.vstack(preds).squeeze().numpy()
            ground_truth = torch.vstack(ground_truth).squeeze().numpy()
            loss_val = loss_val / len(val_loader.dataset)
            acc_val = self.metric(ground_truth, preds)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)

            # test
            preds = []
            ground_truth = []
            loss_test = 0
            for batch in test_loader:
                batch = batch.to(self.device)
                with torch.no_grad():
                    out, _ = self.model(batch)
                    pred = F.softmax(out, dim=1)
                    if self.conf.training['metric'] != 'acc':
                        pred = pred[:, 1].unsqueeze(1)
                    preds.append(pred.detach().cpu())
                    ground_truth.append(batch.y.detach().cpu().unsqueeze(1))
                loss_test += self.loss_fn(out, batch.y.view(-1), reduction='sum').item()
            preds = torch.vstack(preds).squeeze().numpy()
            ground_truth = torch.vstack(ground_truth).squeeze().numpy()
            loss_test = loss_test / len(test_loader.dataset)
            acc_test = self.metric(ground_truth, preds)

            # save
            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop:
                break

            if debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | Acc(test) {:.4f}| {}".format(epoch + 1, time_train, loss_train, acc_train, loss_val, acc_val, acc_test, improve))

        print('Optimization Finished!')
        # print('Time(s): {:.4f}'.format(self.total_time))
        print('Mean Time(s): {:.4f}'.format(np.mean(time_train_list)))
        # test
        preds = []
        ground_truth = []
        self.model.load_state_dict(self.weights)
        self.model.eval()
        loss_test = 0
        for batch in test_loader:
            batch = batch.to(self.device)
            with torch.no_grad():
                out, _ = self.model(batch)
                pred = F.softmax(out, dim=1)
                if self.conf.training['metric'] != 'acc':
                    pred = pred[:,1].unsqueeze(1)
                preds.append(pred.detach().cpu())
                ground_truth.append(batch.y.detach().cpu().unsqueeze(1))
            loss_test += self.loss_fn(out, batch.y.view(-1), reduction='sum').item()
        preds = torch.vstack(preds).squeeze().numpy()
        ground_truth = torch.vstack(ground_truth).squeeze().numpy()
        loss_test = loss_test / len(test_loader.dataset)
        acc_test = self.metric(ground_truth, preds)
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test, acc_test))
        # for tune
        if self.current_split == 0 and os.path.basename(sys.argv[0])[:4] == 'tune':
            if acc_test < self.conf.tune['expect']:
                raise ValueError

        if 'mode' in self.conf and self.conf.mode == 'pretrain':
            self.result = {}
            self.recoder = Recorder(self.conf.training_sota['patience'], self.conf.training_sota['criterion'])
            if 'use_sota' in self.conf:
                if self.conf.use_sota == 'gmt':
                    self.model.backbone = GMT(**self.conf.backbone, n_feat=self.dim_feats, n_class=self.n_classes).to(self.device)
                if self.conf.use_sota == 'topkpool':
                    self.model.backbone = TopKPool(**self.conf.backbone, n_feat=self.dim_feats, n_class=self.n_classes).to(self.device)
                if self.conf.use_sota == 'edgepool':
                    self.model.backbone = EdgePool(**self.conf.backbone, n_feat=self.dim_feats,
                                                   n_class=self.n_classes).to(self.device)

            self.model.backbone.reset_parameters()
            self.model.parsing.requires_grad_(False)
            self.model.selecting.requires_grad_(False)
            self.model.gsl.requires_grad_(False)
            self.model.motif.requires_grad_(False)
            optim = torch.optim.Adam(self.model.backbone.parameters(), lr=self.conf.training_sota['lr'], weight_decay=self.conf.training_sota['weight_decay'])

            if isinstance(self.dataset.data_raw, PygDataset):
                train_dataset = self.dataset.data_raw[self.train_mask]
                test_dataset = self.dataset.data_raw[self.test_mask]
                val_dataset = self.dataset.data_raw[self.val_mask]
            elif isinstance(self.dataset.data_raw, list):
                train_dataset = [self.dataset.data_raw[idx] for idx in self.train_mask.tolist()]
                test_dataset = [self.dataset.data_raw[idx] for idx in self.test_mask.tolist()]
                val_dataset = [self.dataset.data_raw[idx] for idx in self.val_mask.tolist()]
            else:
                raise NotImplementedError

            train_loader = DataLoader(train_dataset, self.conf.training_sota['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, self.conf.training_sota['test_batch_size'], shuffle=False)
            test_loader = DataLoader(test_dataset, self.conf.training_sota['test_batch_size'], shuffle=False)

            for epoch in range(self.conf.training_sota['n_epochs']):

                improve = ''
                t0 = time.time()
                loss_train = 0

                # forward and backward
                preds = []
                ground_truth = []
                self.model.train()
                for batch in train_loader:
                    # print(batch)
                    optim.zero_grad()
                    batch = batch.to(self.device)
                    out, loss_con = self.model(batch)
                    loss = self.loss_fn(out, batch.y.view(-1))
                    loss.backward()
                    optim.step()
                    loss_train += loss.item() * batch.num_graphs
                    pred = F.softmax(out, dim=1)
                    if self.conf.training_sota['metric'] != 'acc':
                        pred = pred[:, 1].unsqueeze(1)
                    preds.append(pred.detach().cpu())
                    ground_truth.append(batch.y.detach().cpu().unsqueeze(1))
                preds = torch.vstack(preds).squeeze().numpy()
                ground_truth = torch.vstack(ground_truth).squeeze().numpy()
                loss_train = loss_train / len(train_loader.dataset)
                acc_train = self.metric(ground_truth, preds)

                # Evaluate
                preds = []
                ground_truth = []
                self.model.eval()
                loss_val = 0
                for batch in val_loader:
                    batch = batch.to(self.device)
                    with torch.no_grad():
                        out, _ = self.model(batch)
                        pred = F.softmax(out, dim=1)
                        if self.conf.training_sota['metric'] != 'acc':
                            pred = pred[:, 1].unsqueeze(1)
                        preds.append(pred.detach().cpu())
                        ground_truth.append(batch.y.detach().cpu().unsqueeze(1))
                    loss_val += self.loss_fn(out, batch.y.view(-1), reduction='sum').item()
                preds = torch.vstack(preds).squeeze().numpy()
                ground_truth = torch.vstack(ground_truth).squeeze().numpy()
                loss_val = loss_val / len(val_loader.dataset)
                acc_val = self.metric(ground_truth, preds)
                flag, flag_earlystop = self.recoder.add(loss_val, acc_val)

                # save
                if flag:
                    improve = '*'
                    self.total_time = time.time() - self.start_time
                    self.best_val_loss = loss_val
                    self.result['valid'] = acc_val
                    self.result['train'] = acc_train
                    self.weights = deepcopy(self.model.state_dict())
                elif flag_earlystop:
                    break

                if debug:
                    print(
                        "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                            epoch + 1, time.time() - t0, loss_train, acc_train, loss_val, acc_val, improve))
            print('Optimization Finished!')
            print('Time(s): {:.4f}'.format(self.total_time))
            # test
            preds = []
            ground_truth = []
            self.model.load_state_dict(self.weights)
            self.model.eval()
            loss_test = 0
            for batch in test_loader:
                batch = batch.to(self.device)
                with torch.no_grad():
                    out, _ = self.model(batch)
                    pred = F.softmax(out, dim=1)
                    if self.conf.training_sota['metric'] != 'acc':
                        pred = pred[:, 1].unsqueeze(1)
                    preds.append(pred.detach().cpu())
                    ground_truth.append(batch.y.detach().cpu().unsqueeze(1))
                loss_test += self.loss_fn(out, batch.y.view(-1), reduction='sum').item()
            preds = torch.vstack(preds).squeeze().numpy()
            ground_truth = torch.vstack(ground_truth).squeeze().numpy()
            loss_test = loss_test / len(test_loader.dataset)
            acc_test = self.metric(ground_truth, preds)
            self.result['test'] = acc_test
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test, acc_test))

        return self.result, None

    def set_method(self):
        # self.model.backbone = GNNEncoder(n_feat=self.dim_feats, n_class=self.n_classes, **self.conf.backbone).to(self.device)
        self.model.reset_parameters()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'], weight_decay=self.conf.training['weight_decay'])

    def init_motif(self, dataset):
        train_loader = DataLoader(dataset, self.conf.training['test_batch_size'], shuffle=True)
        zs = torch.tensor([], device=self.device)
        ys = torch.tensor([], dtype=torch.long, device=self.device)
        self.model.eval()
        for data in train_loader:
            data = data.to(self.device)
            with torch.no_grad():
                out, _, z_sub, belonging, importance = self.model.forward_wo_gsl(data, return_sub=True)
            # 先得到子图的标签
            y_sub = data.y[belonging]
            if self.pretrain_flag:
                # 如果进行了pretrain，可以进行筛选
                logits = F.softmax(out, dim=-1)
                con_max, cls_max = logits.max(dim=1)
                confidence = cls_max[belonging]
                score = (importance + confidence) / 2
                # score = confidence
                rank = score.sort(descending=True)[1]
                con_subgraph_idx = rank[:int(rank.shape[0] * self.conf.motif['k1'])]
                zs = torch.cat([zs, z_sub[con_subgraph_idx]], dim=0)
                ys = torch.cat([ys, y_sub[con_subgraph_idx]], dim=0)
            else:
                zs = torch.cat([zs, z_sub], dim=0)
                ys = torch.cat([ys, y_sub], dim=0)
        self.model.motif.init_motif(zs, ys)
        torch.cuda.empty_cache()

    def update_motif(self, dataset):
        train_loader = DataLoader(dataset, self.conf.training['batch_size'], shuffle=True)
        zs = torch.tensor([], device=self.device)
        ys = torch.tensor([], dtype=torch.long, device=self.device)
        self.model.eval()
        for data in train_loader:
            data = data.to(self.device)
            with torch.no_grad():
                out, _, z_sub, belonging, importance = self.model(data, return_sub=True)
            # 先得到子图的标签
            y_sub = data.y[belonging]
            # 筛选confidence足够大的图的子图
            logits = F.softmax(out, dim=-1)
            con_max, cls_max = logits.max(dim=1)
            confidence = cls_max[belonging]
            score = (importance + confidence) / 2
            # score = confidence
            rank = score.sort(descending=True)[1]
            con_subgraph_idx = rank[:int(rank.shape[0] * self.conf.motif['k1'])]
            zs = torch.cat([zs, z_sub[con_subgraph_idx]], dim=0)
            ys = torch.cat([ys, y_sub[con_subgraph_idx]], dim=0)
        self.model.motif.update_motif(zs, ys)
        torch.cuda.empty_cache()

    def pretrain(self, debug=False):
        if debug:
            print('Pretraining Start')
        if isinstance(self.dataset.data_raw, PygDataset):
            train_dataset = self.dataset.data_raw[self.train_mask]
            test_dataset = self.dataset.data_raw[self.test_mask]
            val_dataset = self.dataset.data_raw[self.val_mask]
        elif isinstance(self.dataset.data_raw, list):
            train_dataset = [self.dataset.data_raw[idx] for idx in self.train_mask.tolist()]
            test_dataset = [self.dataset.data_raw[idx] for idx in self.test_mask.tolist()]
            val_dataset = [self.dataset.data_raw[idx] for idx in self.val_mask.tolist()]

        train_loader = DataLoader(train_dataset, self.conf.training['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, self.conf.training['test_batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, self.conf.training['test_batch_size'], shuffle=False)

        pretrain_backbone = 'pretrain_backbone' in self.conf.training and self.conf.training['pretrain_backbone']
        time_train_list = []
        for epoch in range(self.conf.training['n_pretrain']):
            improve = ''
            t0 = time.time()
            loss_train = 0

            # forward and backward
            preds = []
            ground_truth = []
            self.model.train()
            for data in train_loader:
                # print(data)
                # t1 = time.time()
                self.optim.zero_grad()
                data = data.to(self.device)
                out, _ = self.model.forward_wo_gsl(data, only_backbone=pretrain_backbone)
                loss = self.loss_fn(out, data.y.view(-1))
                loss.backward()
                self.optim.step()
                loss_train += loss.item() * data.num_graphs
                pred = F.softmax(out, dim=1)
                if self.conf.training['metric'] != 'acc':
                    pred = pred[:, 1].unsqueeze(1)
                preds.append(pred.detach().cpu())
                ground_truth.append(data.y.detach().cpu().unsqueeze(1))
                # print(time.time()-t1)
            preds = torch.vstack(preds).squeeze().numpy()
            ground_truth = torch.vstack(ground_truth).squeeze().numpy()
            loss_train = loss_train / len(train_loader.dataset)
            acc_train = self.metric(ground_truth, preds)
            time_train = time.time() - t0
            time_train_list.append(time_train)

            # Evaluate
            preds = []
            ground_truth = []
            self.model.eval()
            loss_val = 0
            for data in val_loader:
                data = data.to(self.device)
                with torch.no_grad():
                    out, _ = self.model.forward_wo_gsl(data, only_backbone=pretrain_backbone)
                    pred = F.softmax(out, dim=1)
                    if self.conf.training['metric'] != 'acc':
                        pred = pred[:, 1].unsqueeze(1)
                    preds.append(pred.detach().cpu())
                    ground_truth.append(data.y.detach().cpu().unsqueeze(1))
                loss_val += self.loss_fn(out, data.y.view(-1), reduction='sum').item()
            preds = torch.vstack(preds).squeeze().numpy()
            ground_truth = torch.vstack(ground_truth).squeeze().numpy()
            loss_val = loss_val / len(val_loader.dataset)
            acc_val = self.metric(ground_truth, preds)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)

            # test
            preds = []
            ground_truth = []
            loss_test = 0
            for batch in test_loader:
                batch = batch.to(self.device)
                with torch.no_grad():
                    out, _ = self.model.forward_wo_gsl(batch, only_backbone=pretrain_backbone)
                    pred = F.softmax(out, dim=1)
                    if self.conf.training['metric'] != 'acc':
                        pred = pred[:, 1].unsqueeze(1)
                    preds.append(pred.detach().cpu())
                    ground_truth.append(batch.y.detach().cpu().unsqueeze(1))
                loss_test += self.loss_fn(out, batch.y.view(-1), reduction='sum').item()
            preds = torch.vstack(preds).squeeze().numpy()
            ground_truth = torch.vstack(ground_truth).squeeze().numpy()
            loss_test = loss_test / len(test_loader.dataset)
            acc_test = self.metric(ground_truth, preds)

            # save
            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop:
                break

            if debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | Acc(test) {:.4f}| {}".format(epoch + 1, time_train, loss_train, acc_train, loss_val, acc_val, acc_test, improve))
        print('Pretrain End')
        # test
        print('Mean Time(s): {:.4f}'.format(np.mean(time_train_list)))
        print(torch.cuda.max_memory_allocated() / 1048576)
        preds = []
        ground_truth = []
        self.model.load_state_dict(self.weights)
        self.model.eval()
        loss_test = 0
        for data in test_loader:
            data = data.to(self.device)
            with torch.no_grad():
                out, _ = self.model.forward_wo_gsl(data, only_backbone=pretrain_backbone)
                pred = F.softmax(out, dim=1)
                if self.conf.training['metric'] != 'acc':
                    pred = pred[:, 1].unsqueeze(1)
                preds.append(pred.detach().cpu())
                ground_truth.append(data.y.detach().cpu().unsqueeze(1))
            loss_test += self.loss_fn(out, data.y.view(-1), reduction='sum').item()
        preds = torch.vstack(preds).squeeze().numpy()
        ground_truth = torch.vstack(ground_truth).squeeze().numpy()
        loss_test = loss_test / len(test_loader.dataset)
        acc_test = self.metric(ground_truth, preds)
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test, acc_test))
        if self.current_split == 0 and os.path.basename(sys.argv[0])[:4] == 'tune':
            if acc_test < self.conf.tune['expect']:
                print('Below Expectation')
                raise ValueError
        torch.cuda.empty_cache()