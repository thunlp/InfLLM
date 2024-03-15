import math
import torch
from .utils import get_mq_attn
from typing import Optional
from copy import deepcopy

class TransferingTensor:
    def __init__(self, tensor, to_cpu: bool):
        if isinstance(tensor, TransferingTensor):
            tensor = tensor.get()

        assert isinstance(tensor, torch.Tensor)
            
        if to_cpu:
            assert tensor.is_cuda
            self.is_cuda = False
        else:
            assert not tensor.is_cuda
            self.is_cuda = True
        
        self.tensor = tensor.to(
            device="cpu" if to_cpu else "cuda",
            non_blocking=True
        )
        self.event = torch.cuda.Event()
        self.event.record()
        self.wait = False

    def get(self):
        if not self.wait:
            self.event.wait()
            self.wait = True
        return self.tensor

    def __len__(self):
        return len(self.tensor)


    def __getattr__(self, name):
        if not self.wait:
            self.event.wait()
            self.wait = True
        return getattr(self.tensor, name)


class VectorTensor:
    def __init__(
        self, 
        element_shape,
        element_dtype,
        concat_dim,
    ):
        init_cached_size = 64
        if concat_dim != 0:
            element_shape = (element_shape[concat_dim],) + element_shape[1:concat_dim] + (element_shape[0],)  + element_shape[concat_dim+1:]
        init_data_shape = (init_cached_size,) + element_shape[1:]
        self.concat_dim = concat_dim
        self.data = torch.empty(
            init_data_shape,
            dtype=element_dtype,
            device='cuda'
        )
        self.length = 0
        self.cache_size = init_cached_size

    def append_cache(self):
        new_cache_size = self.cache_size * 2
        data_shape = self.data.shape
        new_data = torch.empty(
            (new_cache_size,) + data_shape[1:],
            device='cuda',
            dtype=self.data.dtype
        )
        new_data[:self.cache_size,...].copy_(self.data)
        self.data = new_data
        self.cache_size = new_cache_size

    def append(self, tensor: torch.Tensor):
        assert tensor.dtype == self.data.dtype
        assert tensor.is_contiguous()
        if self.concat_dim != 0:
            tensor = tensor.transpose(0, self.concat_dim)

        append_l = tensor.size(0)

        while self.length + append_l > self.cache_size:
            self.append_cache()

        self.data[self.length: self.length+append_l, ...].copy_(tensor)

        self.length += append_l


    def get_data(self):
        if self.concat_dim == 0:
            return self.data[:self.length, ...]

        return self.data[:self.length,...].transpose(0, self.concat_dim)


    def __len__(self):
        return self.length



class ContextManager:
    def __init__(self, 
                 position_embedding,
                 n_init, n_local, 
                 block_size, max_cached_block, topk, exc_block_size, 
                 perhead = False,
                 score_decay: float = 0.1, fattn: bool = False,
                 repr_topk: int = 1,
                 max_calc_block: Optional[int] = None,
                 use_buffer = True,
                 cache_strategy = "lru",
                 calc_block_score = False
    ):
        if max_calc_block is None:
            max_calc_block = topk

        assert max_calc_block >= topk
        self.max_calc_block = max_calc_block
        self.length = 0
        self.position_embedding = position_embedding
        self.n_init = n_init
        self.n_local = n_local
        self.block_size = block_size
        self.max_cached_block = max_cached_block
        self.exc_block_size = exc_block_size
        self.score_decay = score_decay
        assert exc_block_size <= n_local # no global token in input
        self.topk = topk
        self.mq_attn, self.triton_fattn = get_mq_attn(fattn)
        self.fattn = fattn
        self.initialized = False
        self.perhead = perhead
        self.repr_topk = repr_topk
        self.use_buffer = use_buffer
        self.cache_strategy = cache_strategy
        self.calc_block_score = calc_block_score
        self.load_count = 0

        assert cache_strategy in ["lru", "fifo", "lru-s"]

        if cache_strategy == "lru-s":
            assert calc_block_score, "Block score calcualtion is needed for LRU-S cache strategy."

        
    def load_block(self, b, i):
        if i in self.cached_blocks[b]:
            if self.cache_strategy == "lru":
                self.cached_blocks[b][i] = self.load_count
                self.load_count += 1
            assert self.global_blocks[b][i][0].is_cuda
            assert self.global_blocks[b][i][1].is_cuda
            return False

        self.global_blocks[b][i] = (
            TransferingTensor(self.global_blocks[b][i][0], False),
            TransferingTensor(self.global_blocks[b][i][1], False)
        )
        if self.cache_strategy in ["fifo", "lru"]:
            self.cached_blocks[b][i] = self.load_count
            self.load_count += 1
        else:
            self.cached_blocks[b][i] = 0

        return True


    def offload_block(self, u, i):
        if i not in self.cached_blocks[u]:
            return False
        self.global_blocks[u][i] = (
            TransferingTensor(self.global_blocks[u][i][0], True),
            TransferingTensor(self.global_blocks[u][i][1], True),
        )
        self.cached_blocks[u].pop(i)
        return True


    def remove_lru_blocks(self):
        for u in range(self.num_units):
            if len(self.cached_blocks[u]) <= self.max_cached_block:
                continue

            num_remove = len(self.cached_blocks[u]) - self.max_cached_block
            lst = list(self.cached_blocks[u].items())
            lst.sort(key=lambda x: x[1])

            for i in range(num_remove):
                assert self.offload_block(u, lst[i][0])


    def get_block_k(self, k, score):
        assert isinstance(score, torch.Tensor)
        assert k.dim() >= 2
        assert k.shape[:-1] == score.shape
        assert k.shape[-2] == self.block_size
        score_topk = score.topk(self.repr_topk, dim=-1).indices
        assert score_topk.shape == (self.num_units, self.unit_size, self.repr_topk)
        ret = torch.gather(k, -2, score_topk[:, :, :, None].expand(self.num_units, self.unit_size, self.repr_topk, self.dim_head))
        return ret

    def flat_to_unit(self, tensor):
        assert tensor.size(0) == self.batch_size
        if tensor.size(1) == self.num_heads:
            return tensor.view((self.num_units, self.unit_size) + tuple(tensor.shape[2:]))
        elif tensor.size(1) == self.num_heads_kv:
            tensor = tensor.view((self.batch_size, self.num_heads_kv, 1) + tuple(tensor.shape[2:]))
            shape = list(tensor.shape)
            shape[2] *= self.num_heads // self.num_heads_kv
            tensor = tensor.expand(tuple(shape))
            tensor = tensor.reshape((self.batch_size, self.num_heads) + tuple(shape[3:]))
            return tensor.view((self.num_units, self.unit_size) + tuple(tensor.shape[2:]))
        else:
            raise ValueError

    def from_group_kv(self, tensor):
        if self.perhead:
            return tensor

        assert tensor.dim() == 3
        assert tensor.size(0) == self.num_heads_kv
        if self.num_heads == self.num_heads_kv:
            return tensor
        _, length, dim_head = tensor.shape
        num_group = self.num_heads // self.num_heads_kv
        tensor = tensor.view((self.num_heads_kv, 1, length, dim_head))
        tensor = tensor.expand((self.num_heads_kv, num_group, length, dim_head)).reshape((self.num_heads, length, dim_head))
        return tensor

            
    def to_group_kv(self, tensor):
        if self.perhead:
            return tensor

        assert tensor.dim() == 3
        assert tensor.size(0) == self.num_heads
        if self.num_heads == self.num_heads_kv:
            return tensor

        num_group = self.num_heads // self.num_heads_kv
        _, length, dim_head = tensor.shape
        tensor = tensor.view((self.num_heads_kv, num_group, length, dim_head))
        tensor = tensor[:, 0, :, :].contiguous()
        return tensor

    def init(
        self, 
        local_q, local_k, local_v,
        global_q, global_k, global_v
    ):
        assert local_q.dim() == 4
        batch_size, num_heads, len_q, dim_head = local_q.shape
        num_heads_kv = local_k.size(1)

        for _t in [local_q, local_k, local_v, global_q, global_k, global_v]:
            assert _t.size(0) == batch_size
            assert (_t.size(1) == num_heads or _t.size(1) == num_heads_kv)
            assert _t.size(2) == len_q
            assert _t.size(3) == dim_head
            assert _t.is_cuda


        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.dim_head = dim_head
        if self.perhead:
            self.num_units = batch_size * num_heads
            self.unit_size = 1
        else:
            self.num_units = batch_size
            self.unit_size = num_heads

        self.global_blocks = [[] for _ in range(self.num_units)] # [[(global_k, global_v)]]
        self.cached_blocks = [{} for _ in range(self.num_units)] # [[block_id: block_score]
        self.num_global_block = 0

        self.block_k = VectorTensor(
            (self.num_units, self.unit_size, -1, dim_head), global_k.dtype, 2
        )
        self.local_k = torch.empty((self.num_units, self.unit_size, 0, dim_head), dtype=local_k.dtype, device=local_k.device)
        self.local_v = torch.empty((self.num_units, self.unit_size, 0, dim_head), dtype=local_v.dtype, device=local_v.device)

        self.global_remainder = (
            torch.empty((self.num_units, self.unit_size, 0, dim_head), dtype=global_k.dtype, device=global_k.device),
            torch.empty((self.num_units, self.unit_size, 0, dim_head), dtype=global_v.dtype, device=global_v.device),
        )

        self.global_remainder_local_score = torch.empty((self.num_units, self.unit_size, 0), dtype=global_k.dtype, device=global_k.device)


        self.init_k = torch.empty((self.num_units, self.unit_size, 0, dim_head), dtype=global_k.dtype, device=global_k.device)
        self.init_v = torch.empty((self.num_units, self.unit_size, 0, dim_head), dtype=global_k.dtype, device=global_k.device)
        self.init_exc = False
        self.dtype = local_q.dtype

        if self.use_buffer:
            self.global_buffer = torch.zeros(
                    (2, self.num_units, self.unit_size, self.max_calc_block * self.block_size + self.exc_block_size + self.block_size + self.n_init, dim_head),
                    dtype = global_k.dtype, device=global_k.device
                )
            self.global_buffer_block_id_list = [[-1] * self.max_calc_block for _ in range(self.num_units)]
            self.global_buffer_init_len = 0

        self.initialized = True
    

    def calc_block_topk(
        self, global_h_q
    ):
        if self.num_global_block <= self.topk:
            return [list(range(len(self.global_blocks[0]))) for _ in range(self.num_units)]

        block_k = self.block_k.get_data()
        assert block_k.shape == (self.num_units, self.unit_size, self.num_global_block * self.repr_topk, self.dim_head)

        global_repr_logits = torch.matmul(
            global_h_q, block_k.transpose(-1, -2)
        ) # (num_units, unit_size, len_q, num_global_block * repr_topk)

        block_score = global_repr_logits.view(self.num_units, self.unit_size, -1, self.num_global_block, self.repr_topk)
        block_score = block_score.mean(dim=-1).mean(dim=2).mean(dim=1) 

        assert block_score.shape == (self.num_units, self.num_global_block)
        indices = block_score.topk(self.topk, dim=-1).indices.cpu()
        assert indices.shape == (self.num_units, self.topk)

        ret = []
        for u in range(self.num_units):
            ret.append(indices[u].tolist())

        return ret


    def get_local_mask(
        self, local_h_q, local_h_k
    ):
        len_q = local_h_q.size(-2)
        len_k = local_h_k.size(-2)
        device = local_h_q.device
        dist = torch.arange(
            len_q, device=device
        ).unsqueeze(1) - torch.arange(
            len_k, device=device
        ).unsqueeze(0) + len_k - len_q
        return (dist >= 0) & (dist < self.n_local)

    def get_global_hidden_and_mask(
        self, len_q, block_topk
    ):
        assert len(block_topk) == self.num_units
        global_block_map = [[] for _ in range(self.num_units)]
        global_remainder_len = max(self.global_remainder[0].size(-2) + len_q - self.n_local, 0)
        init_len = self.init_k.size(-2)

        total_len = self.max_calc_block * self.block_size + self.exc_block_size + self.block_size + self.n_init
        non_blocking_copy = True
        
        if self.use_buffer:
            global_h_k = self.global_buffer[0]
            global_h_v = self.global_buffer[1]
        else:
            global_h_k = torch.empty(
                (self.num_units, self.unit_size, total_len, self.dim_head),
                device='cuda', dtype=self.dtype
            )
            global_h_v = torch.zeros_like(global_h_k)

        global_mask = torch.ones(
            (len_q, total_len),
            device='cuda', dtype=torch.bool
        )

        block_num = None
        for u in range(self.num_units):
            block_score = []
            for k, s in self.cached_blocks[u].items():
                if k in block_topk[u]:
                    block_score.append((k, float("inf")))
                else:
                    block_score.append((k, min(s, 1e8)))

            block_score.sort(key=lambda x: x[1], reverse=True)
            block_score = block_score[ :self.max_calc_block]

            if block_num is None:
                block_num = len(block_score)
            else:
                # calc block num should be the same for all units
                assert block_num == len(block_score)
            
            st = 0
            ed = 0
            global_block_map[u] = [-1] * block_num
            if self.use_buffer:
                global_block_map[u] = deepcopy(self.global_buffer_block_id_list[u])


            b_idx_list = [block_score[i][0] for i in range(block_num)]
            for b_idx in b_idx_list:
                if b_idx in global_block_map[u]:
                    continue

                st = -1
                ed = -1
                for j in range(self.max_calc_block):
                    if global_block_map[u][j] == -1 or global_block_map[u][j] not in b_idx_list:
                        st = j * self.block_size
                        ed = st + self.block_size
                        global_block_map[u][j] = b_idx
                        break

                global_h_k[u, :, st:ed, :].copy_(self.from_group_kv(self.global_blocks[u][b_idx][0].get()), non_blocking=non_blocking_copy)
                global_h_v[u, :, st:ed, :].copy_(self.from_group_kv(self.global_blocks[u][b_idx][1].get()), non_blocking=non_blocking_copy)

             


        global_mask[ :, block_num * self.block_size: self.max_calc_block * self.block_size] = False
            

        init_st = self.max_calc_block * self.block_size
        init_ed = init_st + init_len
        if not self.use_buffer or self.global_buffer_init_len < init_len:
            global_h_k[:, :, init_st: init_ed, :].copy_(self.init_k, non_blocking=non_blocking_copy)
            global_h_v[:, :, init_st: init_ed, :].copy_(self.init_v, non_blocking=non_blocking_copy)
        
        rmd_st = total_len - global_remainder_len
        rmd_ed = total_len
        global_h_k[:, :, rmd_st: rmd_ed, :].copy_(self.global_remainder[0][:, :, :global_remainder_len, :], non_blocking=non_blocking_copy)
        global_h_v[:, :, rmd_st: rmd_ed, :].copy_(self.global_remainder[1][:, :, :global_remainder_len, :], non_blocking=non_blocking_copy)

        dist = torch.arange(
            len_q, device='cuda'
        ).unsqueeze(dim=1) - torch.arange(
            global_remainder_len,
            device='cuda'
        ).unsqueeze(dim=0) + self.global_remainder[0].size(-2)
        mask = (dist >= self.n_local)

        global_mask[:, init_ed: rmd_st] = False
        global_mask[:, rmd_st: rmd_ed].copy_(mask, non_blocking=non_blocking_copy)

        if self.use_buffer:
            self.global_buffer_block_id_list = deepcopy(global_block_map)
            self.global_buffer_init_len = init_len

        for u in range(self.num_units):
            assert max(global_block_map[u][block_num:] + [-1]) == -1
            assert min(global_block_map[u][:block_num] + [0]) > -1
            global_block_map[u] = list(global_block_map[u][:block_num])

        # wait for non-blocking copy
        torch.cuda.synchronize()

        return global_h_k, global_h_v, global_mask, global_block_map, block_num

    def calc_result_and_score(
        self,
        local_h_q, local_h_k, local_h_v, local_mask,
        global_h_q, global_h_k, global_h_v, global_mask,
        calc_local_score: bool = True,
        calc_global_score: bool = False,
    ):
        assert local_h_q.shape == global_h_q.shape
        if self.triton_fattn:
            ret = self.mq_attn(
                local_h_q, local_h_k, local_h_v, local_mask,
                global_h_q, global_h_k, global_h_v, global_mask,
                output_M=True, output_logits1=calc_local_score, output_logits2=calc_global_score
            )
            o, M = ret[:2]

            assert M.shape == (self.num_units, self.unit_size, global_h_q.size(-2)), f"{M.shape} is not (1, {self.num_units}, {global_h_q.size(-2)})"
            if calc_global_score:
                global_score = ret[3]
                global_score.sub_(M[:, :, :, None])
                global_score.exp2_()
            else:
                global_score = None


            if calc_local_score:
                local_score = ret[2]
                local_score.sub_(M[:, :, :, None])
                local_score.exp2_()
                local_score.masked_fill_(
                    local_mask[None, None, :, :] == False,
                    0
                )
                assert local_score.shape[-2:] == local_mask.shape

            else:
                local_score = None
            

        else:
            o, local_score, global_score = self.mq_attn(
                local_h_q, local_h_k, local_h_v, local_mask,
                global_h_q, global_h_k, global_h_v, global_mask,
                return_score=True
            )

        return o, local_score, global_score


    def update_block_score(
        self, global_score: torch.FloatTensor, global_block_map, global_block_num
    ):
        if global_score is not None:
            global_score = global_score[:, :, :, :global_block_num * self.block_size].mean(dim=-2)
            assert global_score.shape == (self.num_units, self.unit_size, global_block_num * self.block_size)
            global_score = global_score.view(self.num_units, self.unit_size, global_block_num, self.block_size)
            global_score = global_score.sum(dim=-1).sum(dim=1)
            assert global_score.shape == (self.num_units, global_block_num)
            global_score = global_score.to(device='cpu', non_blocking=False) # (num_units, global_block_num)
            for u in range(self.num_units):
                for k, v in self.cached_blocks[u].items():
                    self.cached_blocks[u][k] = v * self.score_decay
                score = global_score[u].tolist()
                assert len(score) >= len(global_block_map[u])
                for s, i in zip(score, global_block_map[u]):
                    self.cached_blocks[u][i] += s


    def append_global(
        self, global_k, global_v, local_score
    ):

        self.global_remainder = (
            torch.cat((self.global_remainder[0], global_k), dim=-2),
            torch.cat((self.global_remainder[1], global_v), dim=-2),
        )

        self.global_remainder_local_score = torch.cat(
            (self.global_remainder_local_score, 
            torch.zeros(
                    (self.num_units, self.unit_size, global_k.size(-2)),
                    dtype=global_k.dtype, device=global_k.device
                )
            ),
            dim=-1
        )

        global_remainder_len = self.global_remainder[0].size(-2)
        assert local_score.shape[:3] == (self.num_units, self.unit_size, global_k.size(-2))
        local_score = local_score[:, :, :, -global_k.size(-2)-self.n_local:]
        assert local_score.size(-1) <= global_remainder_len
        local_score = local_score.sum(dim=-2)
        self.global_remainder_local_score[:, :, global_remainder_len-local_score.size(-1):global_remainder_len].add_(local_score)
        

        if not self.init_exc and global_remainder_len > self.n_local:
            global_k = self.global_remainder[0]
            global_v = self.global_remainder[1]

            append_init_len = min(
                self.n_init - self.init_k.size(-2),
                global_remainder_len - self.n_local
            )
            self.init_k = torch.cat(
                (self.init_k, global_k[:, :, :append_init_len, :]), dim=-2
            )
            self.init_v = torch.cat(
                (self.init_v, global_v[:, :, :append_init_len, :]), dim=-2
            )
            global_k = global_k[:, :, append_init_len:, :]
            global_v = global_v[:, :, append_init_len:, :]
            global_remainder_len -= append_init_len

            if self.init_k.size(-2) == self.n_init:
                self.init_exc = True

            self.global_remainder = (
                global_k, global_v
            )

            self.global_remainder_local_score = self.global_remainder_local_score[:, :, append_init_len:]


        while global_remainder_len - self.block_size >= self.n_local:
            global_remainder_len -= self.block_size
            for u in range(self.num_units):
                self.global_blocks[u].append((
                    TransferingTensor(self.to_group_kv(self.global_remainder[0][u, :, :self.block_size, :]), True),
                    TransferingTensor(self.to_group_kv(self.global_remainder[1][u, :, :self.block_size, :]), True)
                ))

            global_block_k = self.get_block_k(
                self.global_remainder[0][:, :, :self.block_size, :],
                self.global_remainder_local_score[:, :, :self.block_size]
            )
            assert global_block_k.shape == (self.num_units, self.unit_size, self.repr_topk, self.dim_head)

            self.num_global_block += 1
            self.block_k.append(global_block_k)

            self.global_remainder = (
                self.global_remainder[0][:, :, self.block_size:, :],
                self.global_remainder[1][:, :, self.block_size:, :],
            )

            self.global_remainder_local_score = self.global_remainder_local_score[:, :, self.block_size:]
            assert self.global_remainder_local_score.shape == (self.num_units, self.unit_size, global_remainder_len)

            assert self.global_remainder[0].shape == (self.num_units, self.unit_size, global_remainder_len, self.dim_head)
            assert self.global_remainder[1].shape == (self.num_units, self.unit_size, global_remainder_len, self.dim_head)


    def _append(
        self,
        local_q, local_k, local_v,
        global_q, global_k, global_v,
    ):
        # 1. flat to units

        local_q = self.flat_to_unit(local_q)
        local_k = self.flat_to_unit(local_k)
        local_v = self.flat_to_unit(local_v)
        global_q = self.flat_to_unit(global_q)
        global_k = self.flat_to_unit(global_k)
        global_v = self.flat_to_unit(global_v)

        # 2. append local tensor
        self.local_k = torch.cat((self.local_k, local_k), dim=-2)
        self.local_v = torch.cat((self.local_v, local_v), dim=-2)

        # 3. get local_h_q, local_h_k, local_h_v, local_mask
        local_h_q, local_h_k = self.position_embedding(local_q, self.local_k)
        local_h_v = self.local_v
        local_mask = self.get_local_mask(local_h_q, local_h_k)


        # 4. calc topk global repr k and load cache
        global_h_q = self.position_embedding.apply_rotary_pos_emb_one_angle(
            global_q, self.n_local
        )

        block_topk = self.calc_block_topk(global_h_q)

        for u in range(self.num_units):
            for i in block_topk[u]:
                self.load_block(u, i)


        # 5. get global_h_k, global_h_v, global_mask
        #    Beacuse exc_block_size <= n_local, no global_k, global_v used in global part
        global_h_k, global_h_v, global_mask, global_block_map, global_block_num = self.get_global_hidden_and_mask(local_h_q.size(-2), block_topk)
        
        # 6. calc result
        o, loc_score, glb_score = self.calc_result_and_score(
            local_h_q, local_h_k, local_h_v, local_mask,
            global_h_q, global_h_k, global_h_v, global_mask,
            calc_global_score=self.calc_block_score
        )

        # 7. update global score
        self.update_block_score(glb_score, global_block_map, global_block_num)
        
        # 8. update cache
        self.remove_lru_blocks()

        # 9. append global tensor and update local score
        self.append_global(global_k, global_v, loc_score)

        # 10. update local tensor
        if self.local_k.size(-2) >= self.n_local:
            self.local_k = self.local_k[:, :, -self.n_local:, :]
            self.local_v = self.local_v[:, :, -self.n_local:, :]


        return o.view((self.batch_size, self.num_heads, -1, self.dim_head))


    def append(
        self,
        local_q, local_k, local_v,
        global_q, global_k, global_v,
    ):
        if not self.initialized:
            self.init(
                local_q, local_k, local_v,
                global_q, global_k, global_v
            )

        input_length = local_q.size(-2)
        o = torch.empty_like(local_q)
        for st in range(0, input_length, self.exc_block_size): 
            ed = min(st + self.exc_block_size, input_length)
            o[:, :, st: ed, :].copy_(self._append(
                local_q[:, :, st:ed, :],
                local_k[:, :, st:ed, :],
                local_v[:, :, st:ed, :],
                global_q[:, :, st:ed, :],
                global_k[:, :, st:ed, :],
                global_v[:, :, st:ed, :],
            ))

        self.length += input_length

        return o


    def size(self, *args, **kwargs):
        return self.length
