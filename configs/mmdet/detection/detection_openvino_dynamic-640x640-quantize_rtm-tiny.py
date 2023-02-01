_base_ = ['../_base_/base_openvino_dynamic-640x640.py']

# TODO save in mmrazor's checkpoint

global_qconfig=dict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8',
        bit=8,
        is_symmetry=True,
        is_symmetric_range=True),
    a_qscheme=dict(
        qdtype='quint8',
        bit=8,
        is_symmetry=True,
        averaging_constant=0.1),
)

quantizer=dict(
    _scope_='mmrazor',
    type='mmrazor.OpenVINOQuantizer',
    global_qconfig=global_qconfig,
    tracer=dict(
        type='mmrazor.CustomTracer',
        skipped_methods=[
            'mmdet.models.dense_heads.rtmdet_head.RTMDetHead.predict_by_feat',
            'mmdet.models.dense_heads.rtmdet_head.RTMDetHead.loss_by_feat',
        ]
    )
)

checkpoint="/mnt/cache/caoweihan.p/projects/mmrazor_quant2/rtm-tiny/model_ptq_deploy.pth"#'/mnt/cache/caoweihan.p/projects/mmrazor_quant2/debug/model_ptq_deploy.pth'
