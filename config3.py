_base_ = './mmdetection/configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py'

model =  dict(
    bbox_head=dict(
        num_classes=20
    )
)

dataset_type = 'CocoDataset'  # 数据集类型，这将被用来定义数据集。
data_root = 'split_data/'  # 数据的根路径。
test_dataloader = dict(
    batch_size=16, 
    num_workers=4,
    dataset=dict(
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/'), 
        data_root=data_root
    )
)
test_evaluator = dict(ann_file=data_root + 'annotations/val.json', 
                      outfile_prefix='./work_dirs/config3/test'
                    )
train_dataloader = dict(
    batch_size=16, 
    num_workers=4,
    dataset=dict(
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/'), 
        data_root=data_root
    )
)
val_dataloader = dict(
    batch_size=16, 
    num_workers=4,
    dataset=dict(
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/'), 
        data_root=data_root
    )
)
val_evaluator = dict(ann_file=data_root + 'annotations/val.json')
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ]
)
fp16 = dict(loss_scale='dynamic') # 避免loss过大溢出