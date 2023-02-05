exp_name = 'esrgan_psnr'
scale = 4

# model settings
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='RRDBNet',
        in_channels=3,
        out_channels=3,
        mid_channels=16,
        num_blocks=16,
        growth_channels=7,
        upscale_factor=4),

    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean')
)



# model training and testing settings
train_cfg = None

test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale)
#test_cfg = dict(metrics=['PSNR'], crop_border=scale)

# dataset settings
train_dataset_type = 'SRAnnotationDataset'
val_dataset_type = 'SRFolderDataset'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
        #channel_order='rgb'),  #

#    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='RescaleToZeroOne', keys=['gt']),

    dict(type='CopyValues', src_keys=['gt'], dst_keys=['lq']),

    dict(
        type='RandomBlur',
        params=dict(
            kernel_size=[41],
            kernel_list=['iso', 'aniso'],
            kernel_prob=[0.5, 0.5],
            sigma_x=[0.2, 5],
            sigma_y=[0.2, 5],
            rotate_angle=[-3.1416, 3.1416],
        ),
        keys=['lq'],
    ),

    dict(
        type='RandomResize',
        params=dict(
            resize_mode_prob=[0, 1, 0],  # up, down, keep
            resize_scale=[0.0625, 1],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.]),
        keys=['lq'],
    ),

    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian'],
            noise_prob=[1],
            gaussian_sigma=[0, 25],
            gaussian_gray_noise_prob=0),
        keys=['lq'],
    ),

    dict(
        type='RandomJPEGCompression',
        params=dict(quality=[50, 95]),
        keys=['lq']),

    dict(
        type='RandomResize',
        params=dict(
            target_size=(512, 512),
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.]),
        keys=['lq'],
    ),

    dict(type='Quantize', keys=['lq']),

    dict(
        type='RandomResize',
        params=dict(
            target_size=(128, 128), resize_opt=['area'], resize_prob=[1]),
        keys=['lq'],
    ),


    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),


    dict(type='PairedRandomCrop', gt_patch_size=128),

    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),

    dict(type='RandomRotation', degrees=90, keys=['lq', 'gt']),

    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),


    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['gt_path']),

    dict(type='ImageToTensor', keys=['lq', 'gt']),
]






test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
        #channel_order='rgb'),  #

    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
        #channel_order='rgb'),  #

    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'lq_path']),
    
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]




data = dict(
    workers_per_gpu=8,

    train_dataloader=dict(samples_per_gpu=16, drop_last=True), #32

    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),

    #train_dataloader=dict(samples_per_gpu=8, drop_last=True, persistent_workers=False),
    #val_dataloader=dict(samples_per_gpu=1, persistent_workers=False),
    #test_dataloader=dict(samples_per_gpu=1, persistent_workers=False),


    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='data/train/GT',
            gt_folder='data/train/GT',
            ann_file='data/train_ann.txt',
            pipeline=train_pipeline,
            scale=scale)),

    val=dict(
        type=val_dataset_type,
        lq_folder='data/val/LQ',
        gt_folder='data/val/GT',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'),

    test=dict(
        type=val_dataset_type,
        lq_folder='data/val/LQ',
        gt_folder='data/val/GT',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'))




# optimizer
optimizers = dict(generator=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999)))




# learning policy
#total_iters = 400000
total_iters = 200000

#lr_config = dict(
#    policy='CosineRestart',
#    by_epoch=False,
#    periods=[250000, 250000, 250000, 250000],
#    restart_weights=[1, 1, 1, 1],
#    min_lr=1e-7)



lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[100000, 100000],
    restart_weights=[1, 1],
    min_lr=1e-7)


# GIVEN BASELINE
#lr_config = dict(
#    policy='CosineRestart',
#    by_epoch=False,
#    periods=[150000, 150000],
#    restart_weights=[1, 1],
#    min_lr=1e-7)


#lr_config = dict(
#    policy='Step',
#    by_epoch=False,
#    step=[50000, 100000, 200000, 300000],
#    gamma=0.5)


checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)

#evaluation = dict(interval=1000, save_image=True, gpu_collect=True)
evaluation = dict(interval=2500, save_image=True)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit-sr'))
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from =  None
resume_from = 'work_dirs/esrgan_psnr/iter_50000.pth'
#'work_dirs/esrgan_psnr/iter_245000.pth' 
workflow = [('train', 1)]

'''
# optimizer
optimizers = dict(
    generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)),
    discriminator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

# learning policy
total_iters = 400000

lr_config = dict(
    policy='Step',
    by_epoch=False,
    #step=[50000, 100000, 200000, 300000],
    step=[20000, 40000, 60000, 100000],
    gamma=0.5)

#checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)

#evaluation = dict(interval=5000, save_image=True, gpu_collect=True)
evaluation = dict(interval=1000, save_image=True)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit-sr'))
    ])

visual_config = None


# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'

load_from = 'work_dirs/esrgan_psnr/iter_400000.pth'
#load_from = None

resume_from = None

workflow = [('train', 1)]
'''

# model settings
'''
model = dict(
    type='ESRGAN',

    generator=dict(
        type='RRDBNet',
        in_channels=3,
        out_channels=3,
        mid_channels=16, #64
        num_blocks=16, #23
        growth_channels=7, #32 Number of parameters in the model = 1,698,000
        upscale_factor=4),

    discriminator=dict(type='ModifiedVGG', in_channels=3, mid_channels=16), #64

    pixel_loss=dict(type='L1Loss', loss_weight=1e-2, reduction='mean'),

    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={'34': 1.0},
        vgg_type='vgg19',
        perceptual_weight=1.0,
        style_weight=0,
        norm_img=False),

    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=5e-3,
        real_label_val=1.0,
        fake_label_val=0),

    pretrained=None,
)
'''

