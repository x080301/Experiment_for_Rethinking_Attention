import argparse
from openpoints.utils import EasyConfig

parser = argparse.ArgumentParser('Scene segmentation training/testing')
parser.add_argument('--cfg', type=str, default='cfgs/s3dis/pointmetabase-l.yaml', help='config file')
parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
args, opts = parser.parse_known_args()
cfg = EasyConfig()
cfg.load(args.cfg, recursive=True)
cfg.update(opts)  # overwrite the default arguments in yml

#print(cfg.get('model'))


from openpoints.models.segmentation import base_seg
# from openpoints.models import build_model_from_cfg
#
# model = build_model_from_cfg(cfg.model).to(cfg.rank)

model=base_seg.BaseSeg(**cfg.get('model'))
print(model)
