import os
import zarr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import argparse

from gmflow.gmflow import GMFlow
from utils.flow_viz import save_vis_flow_tofile
from utils.utils import InputPadder
from utils import frame_utils
from gmflow.geometry import forward_backward_consistency_check


def load_images_from_zarr(zarr_path, group_path='data/img'):
    """
    从 Zarr 文件中加载整个图像序列，返回 numpy 数组 [T, C, H, W]
    """
    # 打开根组
    root = zarr.open(zarr_path, mode='r')
    # 遍历子组路径
    arr = root
    for key in group_path.split('/'):
        if key not in arr:
            raise KeyError(f"Group '{group_path}' not found in Zarr at {zarr_path}")
        arr = arr[key]
    # 读取所有数据
    imgs = arr[:]  # 可能为 [T, H, W, C] 或 [T, C, H, W]
    # 如果形状为 [T, H, W, C]，转换为 [T, C, H, W]
    if imgs.ndim == 4 and imgs.shape[-1] in (1, 3):
        imgs = imgs.transpose(0, 3, 1, 2)
    # 若单通道，复制为三通道
    if imgs.ndim == 4 and imgs.shape[1] == 1:
        imgs = np.repeat(imgs, 3, axis=1)
    # 最终应为 [T, 3, H, W]
    if imgs.ndim != 4 or imgs.shape[1] != 3:
        raise ValueError(f"Loaded images shape invalid: {imgs.shape}")
    return imgs

@torch.no_grad()
def inference_on_dir(model,
                     inference_dir,
                     output_path='output',
                     padding_factor=8,
                     inference_size=None,
                     paired_data=False,
                     save_flo_flow=False,
                     attn_splits_list=None,
                     corr_radius_list=None,
                     prop_radius_list=None,
                     pred_bidir_flow=False,
                     fwd_bwd_consistency_check=False):
    """从 Zarr 加载图像序列并执行 GMFlow 推理"""
    model.eval()
    if fwd_bwd_consistency_check:
        assert pred_bidir_flow, "Consistency check requires bidir flow"

    os.makedirs(output_path, exist_ok=True)

    # 1) 从 Zarr 中读取所有帧
    images = load_images_from_zarr(inference_dir, group_path='data/img')
    num_frames = images.shape[0]
    print(f"Loaded {num_frames} frames from Zarr at '{inference_dir}/data/img'")

    if paired_data:
        assert num_frames % 2 == 0, "paired_data mode requires even number of frames"
    stride = 2 if paired_data else 1

    # 2) 遍历帧对进行推理
    for i in range(0, num_frames - 1, stride):
        im1 = torch.from_numpy(images[i]).float().cuda()
        im2 = torch.from_numpy(images[i+1]).float().cuda()

        # pad to multiple
        if inference_size is None:
            padder = InputPadder(im1.shape, padding_factor=padding_factor)
            im1_p, im2_p = padder.pad(im1[None], im2[None])
        else:
            im1_p, im2_p = im1[None], im2[None]

        # resize if needed
        if inference_size is not None:
            ori_size = im1_p.shape[-2:]
            im1_p = F.interpolate(im1_p, size=inference_size, mode='bilinear', align_corners=True)
            im2_p = F.interpolate(im2_p, size=inference_size, mode='bilinear', align_corners=True)

        # forward
        out = model(im1_p, im2_p,
                    attn_splits_list=attn_splits_list,
                    corr_radius_list=corr_radius_list,
                    prop_radius_list=prop_radius_list,
                    pred_bidir_flow=pred_bidir_flow)
        flow_pr = out['flow_preds'][-1]

        # resize flow back
        if inference_size is not None:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear', align_corners=True)
            flow_pr[:,0] *= ori_size[-1] / inference_size[-1]
            flow_pr[:,1] *= ori_size[-2] / inference_size[-2]

        # unpad or convert
        if inference_size is None:
            flow = padder.unpad(flow_pr[0]).permute(1,2,0).cpu().numpy()
        else:
            flow = flow_pr[0].permute(1,2,0).cpu().numpy()

        # save visualization
        vis_path = os.path.join(output_path, f'frame{i:04d}_flow.png')
        save_vis_flow_tofile(flow, vis_path)

        # 双向流处理
        if pred_bidir_flow:
            assert flow_pr.size(0) == 2
            if inference_size is None:
                flow_bwd = padder.unpad(flow_pr[1]).permute(1,2,0).cpu().numpy()
            else:
                flow_bwd = flow_pr[1].permute(1,2,0).cpu().numpy()
            save_vis_flow_tofile(flow_bwd, os.path.join(output_path, f'frame{i:04d}_flow_bwd.png'))

            if fwd_bwd_consistency_check:
                if inference_size is None:
                    f_f = padder.unpad(flow_pr[0]).unsqueeze(0)
                    b_f = padder.unpad(flow_pr[1]).unsqueeze(0)
                else:
                    f_f = flow_pr[0].unsqueeze(0)
                    b_f = flow_pr[1].unsqueeze(0)
                occ_f, occ_b = forward_backward_consistency_check(f_f, b_f)
                Image.fromarray((occ_f[0].cpu().numpy()*255).astype(np.uint8))\
                    .save(os.path.join(output_path, f'frame{i:04d}_occ.png'))
                Image.fromarray((occ_b[0].cpu().numpy()*255).astype(np.uint8))\
                    .save(os.path.join(output_path, f'frame{i:04d}_occ_bwd.png'))

        # 保存 .flo
        if save_flo_flow:
            frame_utils.writeFlow(os.path.join(output_path, f'frame{i:04d}_pred.flo'), flow)


def get_args_parser():
    p = argparse.ArgumentParser()
    # 模型配置
    p.add_argument('--resume', default='pretrained/gmflow_sintel-0c07dcb3.pth')
    p.add_argument('--num_scales', type=int, default=1)
    p.add_argument('--feature_channels', type=int, default=128)
    p.add_argument('--upsample_factor', type=int, default=8)
    p.add_argument('--num_transformer_layers', type=int, default=6)
    p.add_argument('--num_head', type=int, default=1)
    p.add_argument('--attention_type', type=str, default='swin')
    p.add_argument('--ffn_dim_expansion', type=int, default=4)
    p.add_argument('--attn_splits_list', nargs='+', type=int, default=[2])
    p.add_argument('--corr_radius_list', nargs='+', type=int, default=[-1])
    p.add_argument('--prop_radius_list', nargs='+', type=int, default=[-1])
    # 推理参数
    p.add_argument('--inference_dir', type=str, required=True,
                   help='zarr 根目录，自动读取 data/img 数组')
    p.add_argument('--output_path', type=str, default='output')
    p.add_argument('--inference_size', nargs='+', type=int, default=None,
                   help='resize 推理大小 [H W]')
    p.add_argument('--dir_paired_data', action='store_true')
    p.add_argument('--save_flo_flow', action='store_true')
    p.add_argument('--pred_bidir_flow', action='store_true')
    p.add_argument('--fwd_bwd_consistency_check', action='store_true')
    return p

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GMFlow(
        feature_channels=args.feature_channels,
        num_scales=args.num_scales,
        upsample_factor=args.upsample_factor,
        num_head=args.num_head,
        attention_type=args.attention_type,
        ffn_dim_expansion=args.ffn_dim_expansion,
        num_transformer_layers=args.num_transformer_layers
    ).to(device)
    ckpt = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    model.cuda()
    inference_on_dir(
        model,
        inference_dir=args.inference_dir,
        output_path=args.output_path,
        padding_factor=8,
        inference_size=args.inference_size,
        paired_data=args.dir_paired_data,
        save_flo_flow=args.save_flo_flow,
        attn_splits_list=args.attn_splits_list,
        corr_radius_list=args.corr_radius_list,
        prop_radius_list=args.prop_radius_list,
        pred_bidir_flow=args.pred_bidir_flow,
        fwd_bwd_consistency_check=args.fwd_bwd_consistency_check
    )
