import logging
import os
import sys
# fix for Windows
if 'QT_QPA_PLATFORM_PLUGIN_PATH' not in os.environ:
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)

from argparse import ArgumentParser

from gui.cutie.utils.palette import custom_palette

def get_arguments():
    parser = ArgumentParser()
    """
    Priority 1: If a "images" folder exists in the workspace, we will read from that directory
    Priority 2: If --images is specified, we will copy/resize those images to the workspace
    Priority 3: If --video is specified, we will extract the frames to the workspace (in an "images" folder) and read from there

    In any case, if a "masks" folder exists in the workspace, we will use that to initialize the mask
    That way, you can continue annotation from an interrupted run as long as the same workspace is used.
    """
    parser.add_argument('--images', help='Folders containing input images.', default=None)
    parser.add_argument('--video', help='Video file readable by OpenCV.', default=None)
    parser.add_argument('--workspace',
                        help='directory for storing buffered images (if needed) and output masks',
                        default=None)
    parser.add_argument('--num_objects', type=int, default=len(custom_palette)//3-1) # //3 because RGB, -1 because background
    parser.add_argument('--workspace_init_only', action='store_true',
                        help='initialize the workspace and exit')
    parser.add_argument('--no-gl', action='store_true',
                        help='disable OpenGL canvas, use legacy QLabel display')
    parser.add_argument('--internal-size', type=int, default=None,
                        help='override max_internal_size (e.g. 720, 1080)')
    parser.add_argument('--profile', action='store_true',
                        help='enable CUDA step profiler (prints timing every 50 frames)')
    parser.add_argument('--auto-propagate-forward', action='store_true',
                        help='start forward propagation automatically after the GUI opens')
    parser.add_argument('--auto-pause-after', type=float, default=None,
                        help='pause propagation after N seconds while keeping the app open')
    parser.add_argument('--display-skip', type=int, default=None,
                        help='set propagation preview skip count in the GUI at startup')

    args = parser.parse_args()
    return args

if __name__ in "__main__":
    # input arguments
    args = get_arguments()

    # perform slow imports after parsing args
    import torch
    from omegaconf import open_dict
    from hydra import compose, initialize
    from PySide6.QtWidgets import QApplication
    import qdarktheme
    from gui.main_controller import MainController

    # logging
    log = logging.getLogger()

    # getting hydra's config without using its decorator
    initialize(version_base='1.3.2', config_path="gui/cutie/config", job_name="gui")
    cfg = compose(config_name="gui_config")

    # general setup
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    args.device = device
    log.info(f'Using device: {device}')

    # merge arguments into config
    args = vars(args)
    # --no-gl flag overrides use_gl_canvas config
    if args.pop('no_gl', False):
        with open_dict(cfg):
            cfg['use_gl_canvas'] = False
    # --internal-size overrides max_internal_size config
    internal_size = args.pop('internal_size', None)
    if internal_size is not None:
        with open_dict(cfg):
            cfg['max_internal_size'] = internal_size
    # --profile enables CUDA step profiler
    profile = args.pop('profile', False)
    if profile:
        with open_dict(cfg):
            cfg['profile'] = True
    with open_dict(cfg):
        for k, v in args.items():
            assert k not in cfg, f'Argument {k} already exists in config'
            cfg[k] = v

    # start everything
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("auto")
    ex = MainController(cfg)
    if 'workspace_init_only' in cfg and cfg['workspace_init_only']:
        sys.exit(0)
    else:
        sys.exit(app.exec())
