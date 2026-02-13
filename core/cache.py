from numpy import append
import torch
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self,
                 dvc: torch.device,
                 max_num_seqs: int,
                 max_seq_len: int,
                 num_layers: int,
                 num_kv_heads: int,
                 head_dim: int):
        logger.info("Initializing with seqs %d len %d layers %d kv %d head_dim %d", max_num_seqs, max_seq_len, num_layers, num_kv_heads, head_dim)
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        # todo - take dtype at run-time.
        self.pool = torch.empty(
            2, max_num_seqs, num_layers, max_seq_len, num_kv_heads, head_dim).to(device=dvc, dtype=torch.bfloat16)
        self.rids = list(range(max_num_seqs))
        self.lengths = list([0] * max_num_seqs)
    
    def alloc_one(self) -> int | None:
        if len(self.rids) == 0:
            return None 

        rid = self.rids.pop()
        self.lengths[rid] = 0
        self.pool[0][rid].zero_()
        self.pool[1][rid].zero_()

        return rid

    def free_one(self, rid: int):
        self.rids = append(self.rids, rid)
        # maybe we don't need it in both places.
        self.lengths[rid] = 0
        self.pool[0][rid].zero_()
        self.pool[1][rid].zero_()

    def get_kv(self, rid: int, layer_idx: int) -> (torch.Tensor | None, torch.Tensor | None, int):
        length = self.lengths[rid]
        
        if length == 0:
            return None, None, 0

        k = self.pool[0, rid, layer_idx, :length]
        v = self.pool[1, rid, layer_idx, :length]
        
        return k, v, length
            
    # ! this assumes that k_new's shape is [T, num_kv_heads, head_dim]
    def store_kv(self, rid: int, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        # todo - remove after testing
        T, H, D = k_new.shape
        assert k_new.shape == v_new.shape
        assert H == self.num_kv_heads, f"Expected {self.num_kv_heads} KV heads, but got {H} (shape: {k_new.shape})"
        assert D == self.head_dim, f"Expected head_dim={self.head_dim}, but got {D} (shape: {k_new.shape})"
        
        cur_len = self.lengths[rid]
        new_len = cur_len + k_new.shape[0]

        self.pool[0, rid, layer_idx, cur_len : new_len] = k_new
        self.pool[1, rid, layer_idx, cur_len : new_len] = v_new

        self.lengths[rid] = new_len