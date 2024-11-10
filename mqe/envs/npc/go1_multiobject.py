# go1_multiobject.py

from mqe.envs.go1.go1 import Go1
import os
from isaacgym import gymapi
from isaacgym.torch_utils import *
import torch

class Go1MultiObject(Go1):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _prepare_npc(self):
        self.init_state_npc = getattr(self.cfg.init_state, "init_states_npc", [])
        self.asset_npcs = []
        self.npc_names = []
        npc_assets_cfg = getattr(self.cfg.asset, "npc_assets", [])
        for npc_cfg in npc_assets_cfg:
            asset_path_npc = npc_cfg['file_npc'].format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            asset_root_npc = os.path.dirname(asset_path_npc)
            asset_file_npc = os.path.basename(asset_path_npc)
            asset_options_npc = gymapi.AssetOptions()
            asset_options_npc.fix_base_link = npc_cfg.get('fix_npc_base_link', False)
            asset_options_npc.disable_gravity = not npc_cfg.get('npc_gravity', True)
            asset_npc = self.gym.load_asset(self.sim, asset_root_npc, asset_file_npc, asset_options_npc)
            self.asset_npcs.append(asset_npc)
            self.npc_names.append(npc_cfg.get('name_npc', 'npc'))

        init_state_list_npc = []
        for init_state_npc in self.init_state_npc:
            base_init_state_list_npc = init_state_npc.pos + init_state_npc.rot + init_state_npc.lin_vel + init_state_npc.ang_vel
            base_init_state_npc = to_torch(base_init_state_list_npc, device=self.device, requires_grad=False)
            init_state_list_npc.append(base_init_state_npc)
        self.base_init_state_npc = torch.stack(init_state_list_npc, dim=0).repeat(self.num_envs, 1)

    def _create_npc(self, env_handle, env_id):
        npc_handles = []
        for i, asset_npc in enumerate(self.asset_npcs):
            start_pose_npc = gymapi.Transform()
            init_state_npc = self.base_init_state_npc[i]
            start_pose_npc.p = gymapi.Vec3(init_state_npc[0].item(), init_state_npc[1].item(), init_state_npc[2].item())
            start_pose_npc.r = gymapi.Quat(init_state_npc[3].item(), init_state_npc[4].item(), init_state_npc[5].item(), init_state_npc[6].item())
            npc_name = self.npc_names[i]
            npc_handle = self.gym.create_actor(env_handle, asset_npc, start_pose_npc, npc_name, env_id, 0, 0)
            npc_handles.append(npc_handle)
        return npc_handles