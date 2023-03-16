_base_ = ['../_base_/base_openvino_dynamic-800x1344.py']

# TODO save in mmrazor's checkpoint
onnx_config = dict(opset_version=11)

global_qconfig = dict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=True),
    a_qscheme=dict(
        qdtype='quint8', bit=8, is_symmetry=True, averaging_constant=0.1),
)

quantizer = dict(
    _scope_='mmrazor',
    type='mmrazor.OpenVINOQuantizer',
    global_qconfig=global_qconfig,
    tracer=dict(
        type='mmrazor.CustomTracer',
        skipped_methods=[
            'mmdet.models.dense_heads.base_dense_head.BaseDenseHead.'
            'predict_by_feat',
            'mmdet.models.dense_heads.anchor_head.AnchorHead.loss_by_feat',
        ]))

checkpoint = '/nvme/caoweihan.p/projects/ckpt/retina_ptq_deploy.pth'
