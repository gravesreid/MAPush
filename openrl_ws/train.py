
from openrl_ws.utils import make_env, get_args, MATWrapper
from mqe.envs.utils import custom_cfg
from openrl.utils.logger import Logger
from openrl.utils.callbacks.checkpoint_callback import CheckpointCallback
import shutil
import os
import torch

from datetime import datetime

def train(args):
    # clear cache
    torch.cuda.empty_cache()

    # cfg_parser = create_config_parser()
    # cfg = cfg_parser.parse_args()

    start_time = datetime.now()
    start_time_str = start_time.strftime("%m-%d-%H")

    if args.algo == "sppo" or args.algo == "dppo":
        single_agent = True
    else:
        single_agent = False
    
    env, env_cfg = make_env(args, custom_cfg(args), single_agent)
    
    if args.algo == "ppo":
        # or use --config ./openrl_ws/cfgs/ppo.yaml in terminal
        args.lr = 0.0005
        args.critic_lr = 0.0005
        args.episode_length = 200

    if "po" in args.algo:
        from openrl.modules.common import PPONet
        from openrl.runners.common import PPOAgent
        # initilize net and agent
        # set args.num_enemies to 2
        args.num_enemies = 2
        net = PPONet(env, cfg=args, device=args.rl_device)
        #print('args.keys: ', args.__dict__.keys())
        #print('args.num_agents: ', args.num_agents)
        #print('args.num_enemies: ', args.num_enemies)
        #print('args.use_obs_instead_of_state: ', args.use_obs_instead_of_state)
        #print(f'args.hidden_size: {args.hidden_size}')
        #print(f'args.mixer_hidden_dim: {args.mixer_hidden_dim}')
        #print(f'ags.hypernet_hidden_dim: {args.hypernet_hidden_dim}')
        #print(f'args.stacked_frames: {args.stacked_frames}')
        #print(f'args.use_stacked_frames: {args.use_stacked_frames}')
        #print('args', args)

        agent = PPOAgent(net)
        # initilize logger
        logger = Logger(
            cfg=net.cfg,
            project_name="MQE",
            scenario_name=args.task,
            wandb_entity="gravesreid",
            exp_name=args.exp_name,
            log_path="./log",
            use_wandb=args.use_wandb,
            use_tensorboard=args.use_tensorboard,
        )
        run_dir = str(logger.run_dir)

        # add config to log
        if args.task == "go1push_mid" or args.task == "go1multiobject":
            source_folder = "./task/"+args.exp_name+"/"
            target_folder = run_dir + "/task/"
            shutil.copytree(source_folder, target_folder)

        if getattr(args, "checkpoint") is not None:
            if os.path.exists(args.checkpoint):
                agent.load(args.checkpoint)
                print("*******************************************************************************************************")
                print("loaded checkpoint: ", args.checkpoint)
                print("*******************************************************************************************************")
            else:
                print("*******************************************************************************************************")
                print("checkpoint not found: ", args.checkpoint)
                print("-------                  NOT LOADED!           --------------------------------------------------------")
                print("*******************************************************************************************************")
        else:
            print("*******************************************************************************************************")
            print("---------       no checkpoint provided       ----------------------------------------------------------")
            print("*******************************************************************************************************")

        # initilize callback
        callback=CheckpointCallback(save_freq=20000, save_path= run_dir + "/checkpoints", name_prefix="rl_model", save_replay_buffer=False, verbose=2)
        agent.train(
            total_time_steps=args.train_timesteps,
            logger=logger,
            callback=callback,
        )
    else:
        agent.train(total_time_steps=args.train_timesteps)

    # move run_folder to result folder if training mid layer
    if args.task == "go1push_mid" or args.task == "go1multiobject":
        source_folder = run_dir
        target_folder = "./results/"+start_time_str+"_"+args.exp_name+"/"
        shutil.copytree(source_folder, target_folder)

if __name__ == '__main__':
    args = get_args()
    print("start training...")
    train(args)
