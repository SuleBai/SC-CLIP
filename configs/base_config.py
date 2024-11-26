# for ViT-B/16
model = dict(
    type='SCCLIPForSegmentation',
    clip_path='ViT-B/16',
    pre_adjust_idx=8,
    post_adjust_idx=3,
    multi_start_idx=3,
    multi_end_idx=10,
    res_cls=0.3
)

# for ViT-L/14
# model = dict(
#     type='SCCLIPForSegmentation',
#     clip_path='ViT-L/14',
#     pre_adjust_idx=16,
#     post_adjust_idx=6,
#     multi_start_idx=8,
#     multi_end_idx=19,
#     res_cls=0.1
# )

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=1))