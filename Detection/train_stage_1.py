import logging
import warnings
import os, sys, shutil
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.engine import default_argument_parser, launch
from detectron2.modeling import build_model

from trainer.mot_trainer import MOTTrainer
from Detection.data.register_dataset import RegisterDataset
from Detection.data.analyse_dataset import get_dataset_info
from Detection.utils.cfg_setup import init_cfg
from Detection.utils.logging_utils import setup_logging

sys.path.append('..')
from baseconfig import CONF

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')

def setup():
    TRAIN_IMG_DIR = CONF.PATH.DETECTTRAIN
    mapping_list, name_list = get_dataset_info(TRAIN_IMG_DIR)
    mapping_list, name_list = zip(*sorted(zip(mapping_list, name_list)))

    num_classes = len(mapping_list)
    cfg = init_cfg(num_classes)
    return cfg, mapping_list, name_list

def main(args):
    setup_logging()

    cfg, mapping_list, name_list = setup()
    logger.info(f"Existing classes: {name_list}")

    register_dataset = RegisterDataset(mapping_list, name_list)
    register_dataset.register_to_catalog()

    # Visualize annotations for debugging
    # register_dataset.visualize_annotations()

    if os.path.exists(CONF.PATH.DETECTOUTPUT):
        logger.warning('Removing old outputs.')
        shutil.rmtree(CONF.PATH.DETECTOUTPUT)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    model = build_model(cfg)
    if args.eval_only:
        logger.info('Running only evaluation.')
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        return MOTTrainer.do_test(cfg, model, False)

    MOTTrainer.do_train(cfg, model, resume=args.resume)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
