import os
import argparse


parser = argparse.ArgumentParser(description="AML Generic Launcher")
parser.add_argument("--workdir", default="",
                        help="The working directory.")
parser.add_argument("--cxk_volna", default="",
                        help="The working directory.")
# parser.add_argument("--cfg", default="")
args, _ = parser.parse_known_args()

# start training
os.chdir(args.workdir)
# The train.py is in the blob under the args.workdir
# os.system("WORKDIR=%s && python train.py --model_cfg %s --solver_cfg %s --evaluator_cfg %s --cuda" % (args.workdir, model_cfg, solver_cfg, evaluator_cfg))

'''
Change the command you want to execute here
'''
os.environ['cxk_vilna'] = args.cxk_volna        # 设置数据根目录
os.system("sed -i 's/\r//g' run.sh")
os.system("bash run.sh {}".format(args.cxk_volna))
