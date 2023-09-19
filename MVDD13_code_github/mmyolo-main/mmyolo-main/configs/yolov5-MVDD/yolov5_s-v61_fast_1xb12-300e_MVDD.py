_base_ = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

data_root = '/root/autodl-tmp/mmyolo-main/data/COCO/'
class_name = ('cargo', 'tanker', 'bulker', 'containership', 'tug', 'fishing', 'drill', 'passenger','cruise', 'warship', 'sailingboat', 'submarine', 'firefighting' )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette= [(255, 0, 0), (255, 0, 255), (0, 255, 0), (255, 128, 0), (153, 51, 250),
     (0, 0, 255), (255, 255, 0), (255, 192, 203), (0, 199, 140), (218, 112, 214),
     (0, 255, 255), (135, 206, 250), (127, 255, 0)])

anchors = [
    [(68, 69), (154, 91), (143, 162)],  # P3/8
    [(242, 160), (189, 287), (391, 207)],  # P4/16
    [(353, 337), (539, 341), (443, 432)]  # P5/32
]

max_epochs = 300
train_batch_size_per_gpu = 12
train_num_workers = 4

load_from = '/root/autodl-tmp/mmyolo-main/pretrain/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'  # noqa

model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/')))

test_dataloader = val_dataloader
# 在测试集上推理，
# 并将检测结果转换格式以用于提交结果
# test_dataloader = dict(
#     batch_size=8,
#     num_workers=4,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         metainfo=metainfo,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/instances_test2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         # pipeline=test_pipeline))


_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator
# test_evaluator = dict(
#     type='mmdet.CocoMetric',
#     ann_file=data_root + 'annotations/instances_test2017.json',
#     metric='bbox',
#     format_only=False,  # 只将模型输出转换为coco的 JSON 格式并保存
#     classwise=True,
#     outfile_prefix='/root/autodl-tmp/mmyolo-main/yolov5s0913')  # 要保存的 JSON 文件的前缀

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=200, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=1000),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa
