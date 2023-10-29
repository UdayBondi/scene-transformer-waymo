import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.functional as FU
from torchvision import transforms

import sys, os
import os.path as osp
import numpy as np
import cv2
import copy
import hydra
import pytorch_lightning as pl

from model.encoder import Encoder
from model.decoder import Decoder
from datautil.waymo_dataset import xy_to_pixel

COLORS = [(0,0,255), (255,0,255), (180,180,0), (143,143,188), (0,100,0), (128,128,0)]
TrajCOLORS = [(0,0,255), (200,0,0), (200,200,0), (0,200,0)]

class SceneTransformer(pl.LightningModule):
    def __init__(self, cfg):
        super(SceneTransformer, self).__init__()
        self.cfg = cfg
        self.in_feat_dim = cfg.model.in_feat_dim
        self.in_dynamic_rg_dim = cfg.model.in_dynamic_rg_dim
        self.in_static_rg_dim = cfg.model.in_static_rg_dim
        self.time_steps = cfg.model.time_steps
        self.current_step = cfg.model.current_step
        self.feature_dim = cfg.model.feature_dim
        self.head_num = cfg.model.head_num
        self.k = cfg.model.k
        self.F = cfg.model.F

        self.Loss = nn.MSELoss(reduction='none')

        self.encoder = Encoder(self.device, self.in_feat_dim, self.in_dynamic_rg_dim, self.in_static_rg_dim,
                                    self.time_steps, self.feature_dim, self.head_num)
        self.decoder = Decoder(self.device, self.time_steps, self.feature_dim, self.head_num, self.k, self.F)

        ### viz options
        self.width = cfg.viz.width
        self.totensor = transforms.ToTensor()

    def forward(self, states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch,
                    roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch,
                        agent_rg_mask, agent_traffic_mask):


        e = self.encoder(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch,
                                    roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch,
                                        agent_rg_mask, agent_traffic_mask)
        encodings = e['out']
        decoding = self.decoder(encodings, agents_batch_mask, states_padding_mask_batch)

        return {'prediction': decoding.permute(1,2,0,3), 'att_weights': e['att_weights']}

    def training_step(self, batch, batch_idx):

        states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, \
                    roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch, \
                        agent_rg_mask, agent_traffic_mask, (num_agents_accum, num_rg_accum, num_tl_accum), \
                            sdc_masks, center_ps = batch.values()

        # Predict
        out = self(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, 
                        roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch,
                            agent_rg_mask, agent_traffic_mask)
        prediction = out['prediction']

        # Calculate Loss
        # to_predict_mask = ~states_padding_mask_batch*states_hidden_mask_batch   # Calculate loss only on the future time steps
        to_predict_mask = ~states_padding_mask_batch  # Calculate loss on the entire time range
        gt = states_batch[:,:,:2]
        gt = gt[to_predict_mask]
        prediction = prediction[to_predict_mask]    

        loss_ = self.Loss(gt.unsqueeze(1).repeat(1,self.F,1), prediction)
        loss_ = torch.mean(torch.mean(loss_, dim=0),dim=-1) * self.cfg.dataset.halfwidth

        k_ = 1
        loss_, _ = torch.topk(loss_, k_)
        loss_ = torch.mean(loss_)
        self.log_dict({'train/loss':loss_})
    
        return loss_

    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            print(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()

    def validation_step(self, batch, batch_idx):
        
        states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, \
                    roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch, \
                        agent_rg_mask, agent_traffic_mask, (num_agents_accum, num_rg_accum, num_tl_accum), \
                            sdc_masks, center_ps = batch.values()

        # Predict
        out = self(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, 
                        roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch,
                            agent_rg_mask, agent_traffic_mask)
        prediction = out['prediction']

        # Calculate Loss
        # to_predict_mask = ~states_padding_mask_batch*states_hidden_mask_batch
        to_predict_mask = ~states_padding_mask_batch

        gt = states_batch[:,:,:2]
        
        loss_ = self.Loss(gt[to_predict_mask].unsqueeze(1).repeat(1,self.F,1), prediction[to_predict_mask])            # [A*T,candi]
        loss_ = torch.min(torch.mean(torch.mean(loss_, dim=0),dim=-1)) * self.cfg.dataset.halfwidth

        rs_error = ((prediction - gt.unsqueeze(2)) ** 2).sum(dim=-1).sqrt_() * self.cfg.dataset.halfwidth
        rs_error[~to_predict_mask]=0
        rse_sum = rs_error.sum(1)
        ade_mask = to_predict_mask.sum(-1)!=0
        ade = (rse_sum[ade_mask].permute(1,0)/to_predict_mask[ade_mask].sum(-1)).permute(1,0)

        fde_mask = to_predict_mask[:,-1]==True
        fde = rs_error[fde_mask][:,-1,:]
        
        minade, _ = ade.min(dim=-1)
        avgade = ade.mean(dim=-1)
        minfde, _ = fde.min(dim=-1)
        avgfde = fde.mean(dim=-1)

        batch_minade = (minade.sum())/(len(minade)+1e-6)
        batch_minfde = (minfde.sum())/(len(minfde)+1e-6)
        batch_avgade = (avgade.sum())/(len(avgade)+1e-6)
        batch_avgfde = (avgfde.sum())/(len(avgfde)+1e-6)

        self.log_dict({'val/loss': loss_, 'val/minade': batch_minade, 'val/minfde': batch_minfde, 'val/avgade': batch_avgade, 'val/avgfde': batch_avgfde})

        self.val_out =  {'states': states_batch, 'states_padding': states_padding_mask_batch, 'states_hidden': states_hidden_mask_batch, 
                        'roadgraph_feat': roadgraph_feat_batch, 'roadgraph_padding': roadgraph_padding_batch, 
                        'traffic_light_feat': traffic_light_feat_batch, 'traffic_light_padding': traffic_light_padding_batch,
                        'num_agents_accum': num_agents_accum, 'num_rg_accum': num_rg_accum, 'num_tl_accum': num_tl_accum, 'sdc_masks': sdc_masks, 'center_ps': center_ps,
                        'pred': prediction, 'loss': loss_, 'att_weights': out['att_weights']}

        return loss_

    def validation_epoch_end(self, outputs) -> None:

        states_batch, states_padding_batch, states_hidden_batch, \
                    roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch, \
                        num_agents_accum, num_rg_accum, num_tl_accum, sdc_masks, center_ps, \
                            prediction, loss, att_weights = self.val_out.values()
        total_empty = np.ones((self.width,self.width,3))*255
        current_step=10
        scene_imgs = []

        for ii, i in enumerate(range(len(num_agents_accum)-1)):
            total_empty_ = total_empty.copy()

            states_, states_padding_, states_hidden_ = states_batch[num_agents_accum[i]:num_agents_accum[i+1]].cpu(), states_padding_batch[num_agents_accum[i]:num_agents_accum[i+1]].cpu(), states_hidden_batch[num_agents_accum[i]:num_agents_accum[i+1]].cpu()
            pred_ = prediction[num_agents_accum[i]:num_agents_accum[i+1]].detach().cpu()
            roadgraph_feat_, roadgraph_padding_ = roadgraph_feat_batch[num_rg_accum[i]:num_rg_accum[i+1]].cpu(), roadgraph_padding_batch[num_rg_accum[i]:num_rg_accum[i+1]].cpu()
            roadgraph_type_, roadgraph_id_ = roadgraph_feat_[:,current_step,-2].cpu(), roadgraph_feat_[:,current_step,-1].cpu()
            traffic_light_, traffic_light_padding_ = traffic_light_feat_batch[num_tl_accum[i]:num_tl_accum[i+1]].cpu(), traffic_light_padding_batch[num_tl_accum[i]:num_tl_accum[i+1]].cpu()
            agt_rg_attmp = att_weights[2][current_step][num_agents_accum[i]:num_agents_accum[i+1],num_rg_accum[i]:num_rg_accum[i+1]]
            agt_tl_attmp = att_weights[3][current_step][num_agents_accum[i]:num_agents_accum[i+1],num_tl_accum[i]:num_tl_accum[i+1]]

            center_p = center_ps[i].cpu()
            ctline_mask = (roadgraph_feat_[...,-2]==2)[:,0]  # LaneCenter-SurfaceStreet

            # Road 
            lane_mask = torch.zeros(ctline_mask.shape, dtype=torch.bool)
            for type_ in [6,7,8,9,10,11,12,13,14]:
                lane_mask += (roadgraph_type_==type_)

            for id_ in np.unique(roadgraph_id_[lane_mask]):
                lane_id_mask = lane_mask*(roadgraph_id_==id_)
                lane_xy = roadgraph_feat_[:,current_step,:2][lane_id_mask] - center_p
                lane_xy *= (self.width/2)
                lane_xy = xy_to_pixel(lane_xy,self.width)
                polygon = np.array([lane_xy.numpy()], np.int32)
                cv2.polylines(total_empty_, polygon, isClosed=False, color=(125,125,125), thickness=1)

            ## Draw road 

            for id_ in np.unique(roadgraph_id_[ctline_mask]):
                ctline_id_mask = ctline_mask*(roadgraph_id_==id_)
                ctline_xy = roadgraph_feat_[:,current_step,:2][ctline_id_mask] - center_p
                ctline_xy *= (self.width/2)
                ctline_xy = xy_to_pixel(ctline_xy,self.width)

                polygon = np.array([ctline_xy.numpy()], np.int32)

                cv2.polylines(total_empty_, polygon, isClosed=False, color=(20,20,20), thickness=1)

            for si_, (s_, p_, sp_, sh_) in enumerate(zip(states_, pred_, states_padding_, states_hidden_)):

                s__ = s_[:,:2].clone() * (self.width/2)
                s__ = xy_to_pixel(s__, self.width)
                p__ = p_.clone() * (self.width/2)
                p__ = xy_to_pixel(p__[sh_], self.width)
                curpt_ = s__[current_step].unsqueeze(0).repeat(self.cfg.model.F,1).unsqueeze(0)
                p__ = torch.cat((curpt_, p__), 0).permute(1,0,2)

                if not sp_[current_step]:
                    polygon = np.array([s__[~sp_].numpy()], np.int32)
                    cv2.polylines(total_empty_, polygon, isClosed=False, color=COLORS[si_%len(COLORS)], thickness=2)
                    cv2.circle(img=total_empty_, center=tuple(s__[current_step].numpy().astype(np.int32)), radius=6, color=COLORS[si_%len(COLORS)], thickness=cv2.FILLED)

                    for pi, p___ in enumerate(p__):
                        p___ = p___[~sp_[current_step:]]
                        mask_x = (p___[...,0] <= self.width)*(p___[...,0]>=0)
                        mask_y = (p___[...,1] <= self.width)*(p___[...,1]>=0)
                        p___ = p___[mask_x*mask_y]

                        polygon = np.array([p___.numpy()], np.int32)

                        for polyi in range(polygon.shape[1]): 
                            cv2.circle(total_empty_, center=(polygon[:, polyi, 0].item(), polygon[:, polyi, 1].item()), radius=3, color=COLORS[si_%len(COLORS)], thickness=1)

                else:
                    polygon = np.array([s__[~sp_].numpy()], np.int32)
                    cv2.polylines(total_empty_, polygon, isClosed=False, color=(150,150,150), thickness=2)

            red_mask = ((traffic_light_[:,current_step,-1] == 4) + (traffic_light_[:,current_step,-1] == 7))
            yellow_mask = ((traffic_light_[:,current_step,-1] == 5) + (traffic_light_[:,current_step,-1] == 8))
            green_mask = traffic_light_[:,current_step,-1] == 6

            red_xy = xy_to_pixel(traffic_light_[:,current_step,:2][red_mask].clone() - center_p, self.width).numpy()#*(width/2)
            yellow_xy = xy_to_pixel(traffic_light_[:,current_step,:2][yellow_mask].clone() - center_p, self.width).numpy()#*(width/2)
            green_xy = xy_to_pixel(traffic_light_[:,current_step,:2][green_mask].clone() - center_p, self.width).numpy()#*(width/2)
            total_empty_ = cv2.cvtColor(total_empty_.astype(np.uint8), cv2.COLOR_BGR2RGB)

            scene_imgs.append((self.totensor(total_empty_)).unsqueeze(0))
        scene_imgs = torch.cat(scene_imgs, dim=0)
        self.logger.experiment.add_image('val/viz', scene_imgs, self.global_step, dataformats="NCHW")

        return super().validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, \
                    roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch, \
                        agent_rg_mask, agent_traffic_mask, (num_agents_accum, num_rg_accum, num_tl_accum), \
                            sdc_masks, center_ps = batch.values()
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        # Predict
        prediction = self(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, 
                        roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch,
                            agent_rg_mask, agent_traffic_mask)
        end_time.record()

        predict_time = self.cfg.dataset.test.prediction_time*10 # Convert seconds to steps
        current_step = self.cfg.model.current_step
        to_predict_time = current_step+predict_time+1

        states_hidden_mask_batch[:, to_predict_time:] = False

        prediction = prediction['prediction']
        # Calculate Loss
        to_predict_mask = ~states_padding_mask_batch*states_hidden_mask_batch

        gt = states_batch[:,:,:2]
        av = gt[sdc_masks]

        Loss = nn.MSELoss(reduction='none')
        loss_ = Loss(gt[to_predict_mask].unsqueeze(1).repeat(1,self.F,1), prediction[to_predict_mask])            # [A*T,candi]
        loss_ = torch.mean(torch.mean(loss_, dim=0),dim=-1)
        arg_min = torch.argmin(loss_)
        loss_ = torch.min(loss_)  

        # Visualize
        width = 1000
        empty_pane = np.ones((width,width,3))*255
        roadgraph_type = roadgraph_feat_batch[:,0,-2].detach().cpu().type(torch.int32).numpy()
        roadgraph_id = roadgraph_feat_batch[:,0,-1].detach().cpu().type(torch.int32).numpy()
        mask_ctline = (roadgraph_type==2)

        # Draw LaneCenter-SurfaceStreet
        for id_ in np.unique(roadgraph_id[mask_ctline]):
            ctline_id_mask = mask_ctline*(roadgraph_id==id_)
            xy_ctline = roadgraph_feat_batch[:,0,:2][ctline_id_mask]
            xy_ctline *= int(width/2)
            xy_ctline[:,1] *= -1
            xy_ctline += int(width/2)
            mask_x = (xy_ctline[:,0] <= width)*(xy_ctline[:,0]>=0)
            mask_y = (xy_ctline[:,1] <= width)*(xy_ctline[:,1]>=0)

            xy_ctline = xy_ctline[mask_x*mask_y]

            polygon = np.array([xy_ctline.detach().cpu().numpy()], np.int32)
            cv2.polylines(empty_pane, polygon, isClosed=False, color=(0,0,0), thickness=1)

        # Draw RoadEdgeBoundary
        mask_ctline = (roadgraph_type==15)
        for id_ in np.unique(roadgraph_id[mask_ctline]):
            ctline_id_mask = mask_ctline*(roadgraph_id==id_)
            xy_ctline = roadgraph_feat_batch[:,0,:2][ctline_id_mask]
            xy_ctline *= int(width/2)
            xy_ctline[:,1] *= -1
            xy_ctline += int(width/2)
            mask_x = (xy_ctline[:,0] <= width)*(xy_ctline[:,0]>=0)
            mask_y = (xy_ctline[:,1] <= width)*(xy_ctline[:,1]>=0)

            xy_ctline = xy_ctline[mask_x*mask_y]

            polygon = np.array([xy_ctline.detach().cpu().numpy()], np.int32)
            cv2.polylines(empty_pane, polygon, isClosed=False, color=(26,118,238), thickness=2)

        gt_ = copy.deepcopy(gt)
        gt_ = gt_.detach().cpu().numpy()

        paddings_ = copy.deepcopy(states_padding_mask_batch)
        paddings_ = paddings_.detach().cpu().numpy()


        paddings_[:, to_predict_time:] = True

        prediction_ = copy.deepcopy(prediction)

        prediction_ = prediction_.detach().cpu().numpy()

        predmask_ = copy.deepcopy(to_predict_mask)
        predmask_ = predmask_.detach().cpu().numpy()

        # Draw circle around AV
        cv2.circle(img=empty_pane, center=(width//2, width//2), radius=20, color=(75,0,130), thickness=2)

        for i, xy_ in enumerate(gt_):
            xy_ *= int(width/2)
            xy_[:,1] *= -1
            xy_ += int(width/2)
            
            xy_past_, xy_future_ = xy_[:self.cfg.model.current_step,:], xy_[self.cfg.model.current_step+1:,:]
            xy_past_padding = paddings_[i,:self.cfg.model.current_step]
            xy_future_padding = paddings_[i,self.cfg.model.current_step+1:]

            xy_current_ = xy_[self.cfg.model.current_step,:]
            # xy_current_padding = paddings_[i, self.cfg.model.current_step]
            

            xy_past_, xy_future_ = xy_past_[~xy_past_padding], xy_future_[~xy_future_padding]
            xy_current_ = np.array(xy_current_, np.int32)

            rect_size = 8
            start = ((xy_current_[0]-rect_size, xy_current_[1]+rect_size))
            end = (xy_current_[0]+rect_size, xy_current_[1]-rect_size)

            cv2.rectangle(empty_pane, start, end, color=COLORS[i%len(COLORS)], thickness=-1)

            # Draw the future ground truth
            polygon = np.array([xy_future_], np.int32)
            cv2.polylines(empty_pane, polygon, isClosed=False, color=COLORS[i%len(COLORS)], thickness=4)

            # draw prediction
            pred_ = prediction_[i]
            pmask_ = predmask_[i]

            predxy_ = copy.deepcopy(pred_[:,arg_min,:])
            predxy_ = predxy_[pmask_]

            predxy_ *= int(width/2)
            predxy_[:,1] *= -1
            predxy_ += int(width/2)
            polygon = np.array([predxy_], np.int32)

            for polyi in range(polygon.shape[1]): 
                cv2.circle(img=empty_pane, center=(polygon[:, polyi, 0].item(), polygon[:, polyi, 1].item()), radius=3, color=COLORS[i%len(COLORS)], thickness=1)

        rs_error = ((prediction - gt.unsqueeze(2)) ** 2).sum(dim=-1).sqrt_()*self.cfg.dataset.halfwidth
        rs_error[~to_predict_mask]=0
        rse_sum = rs_error.sum(1)
        ade_mask = to_predict_mask.sum(-1)!=0
        ade = (rse_sum[ade_mask].permute(1,0)/to_predict_mask[ade_mask].sum(-1)).permute(1,0)

        fde_mask = to_predict_mask[:,to_predict_time-1]==True
        fde = rs_error[fde_mask][:,to_predict_time-1,:]

        minade, _ = ade.min(dim=-1)
        avgade = ade.mean(dim=-1)
        minfde, _ = fde.min(dim=-1)
        avgfde = fde.mean(dim=-1)

        batch_minade = (minade.sum())/(len(minade)+1e-6)
        batch_minfde = (minfde.sum())/(len(minfde)+1e-6)
        batch_avgade = (avgade.sum())/(len(avgade)+1e-6)
        batch_avgfde = (avgfde.sum())/(len(avgfde)+1e-6)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (50,50)
        fontScale              = 0.5
        fontColor              = (0,0,0)
        thickness              = 1
        lineType               = 2

        cv2.putText(empty_pane,"({2}secs)-> minADE:{0} || minFDE:{1}".format(round(batch_minade.item(),2), round(batch_minfde.item(),2), predict_time/10), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

        imgnm = os.path.join(os.getcwd(),'results',f'{batch_idx}_{i}.jpg')
        cv2.imwrite(imgnm, empty_pane)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        time_s = start_time.elapsed_time(end_time)/1000
        # print(time_s)

        self.log_dict({'test/loss': loss_, 'test/minade': batch_minade, 'test/minfde': batch_minfde, 'test/avgade': batch_avgade, 'test/avgfde': batch_avgfde, 'test/time_sec': time_s})

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9,0.999))

        return optimizer
