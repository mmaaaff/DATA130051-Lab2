_base_ = './mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

default_hooks = dict(
    checkpoint=dict(interval=3, type='CheckpointHook'),
)

model = dict(
    backbone=dict(
        frozen_stages=0,
        norm_eval=False
    ), 
    roi_head=dict(
        bbox_head=dict(
            num_classes=20
        )
    ),
    test_cfg=dict(
        rcnn=dict(
            nms=dict(iou_threshold=0.4, type='nms'),
            score_thr=0.5
        )
    )
)

dataset_type = 'CocoDataset'  # 数据集类型，这将被用来定义数据集。
data_root = 'split_data/'  # 数据的根路径。
test_dataloader = dict(
    batch_size=4, 
    num_workers=4,
    dataset=dict(
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/'), 
        data_root=data_root
    )
)
test_evaluator = dict(ann_file=data_root + 'annotations/val.json', 
                      outfile_prefix='./work_dirs/config1/test'
                    )
train_cfg = dict(max_epochs=30, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=4, 
    num_workers=4,
    dataset=dict(
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/'), 
        data_root=data_root
    )
)
val_dataloader = dict(
    batch_size=4, 
    num_workers=4,
    dataset=dict(
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/'), 
        data_root=data_root
    )
)
val_evaluator = dict(ann_file=data_root + 'annotations/val.json')

param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=30,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ]
)
fp16 = dict(loss_scale='dynamic') # 避免loss过大溢出