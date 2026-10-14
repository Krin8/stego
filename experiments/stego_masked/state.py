"""Complete CPU state for verified serialization; production runs never resume."""
import random
import numpy as np
import torch

def capture(model, optimizer, rng, order, cursor, step, bad):
    ns=np.random.get_state()
    return dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
      torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
      numpy_rng=dict(kind=ns[0],keys=torch.from_numpy(ns[1].astype(np.int64)),pos=ns[2],has_gauss=ns[3],cached_gaussian=ns[4]),
      generator_rng=rng.bit_generator.state,order=torch.from_numpy(order.copy()),cursor=cursor,
      step=step,bad_checks=bad,model_training=model.training,threads=torch.get_num_threads())

def restore(state,model,optimizer,rng):
    assert torch.get_num_threads()==state['threads']
    model.load_state_dict(state['state_dict'],strict=True);optimizer.load_state_dict(state['optimizer'])
    model.train(state['model_training']);torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng'])
    ns=state['numpy_rng'];np.random.set_state((ns['kind'],ns['keys'].numpy().astype(np.uint32),ns['pos'],ns['has_gauss'],ns['cached_gaussian']))
    rng.bit_generator.state=state['generator_rng']
    return state['order'].numpy().copy(),state['cursor'],state['step'],state['bad_checks']
