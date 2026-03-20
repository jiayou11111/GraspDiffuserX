import torch
import torch.nn as nn

class DictOfTensorMixin(nn.Module):
    def __init__(self, params_dict=None):
        super().__init__()
        if params_dict is None:
            params_dict = nn.ModuleDict()
        self.params_dict = params_dict

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        class DynamicModule(nn.Module):
            def __getitem__(self, key):
                if key in self._modules:
                    return self._modules[key]
                if key in self._parameters:
                    return self._parameters[key]
                raise KeyError(key)
            
            def __setitem__(self, key, value):
                if isinstance(value, nn.Module):
                    self.add_module(key, value)
                elif isinstance(value, (torch.Tensor, nn.Parameter)):
                    if not isinstance(value, nn.Parameter):
                        value = nn.Parameter(value)
                    self.register_parameter(key, value)
                else:
                    raise TypeError(f"Unsupported type {type(value)}")
                    
            def keys(self):
                return list(self._modules.keys()) + list(self._parameters.keys())

            def values(self):
                 return [self[k] for k in self.keys()]

            def items(self):
                 return [(k, self[k]) for k in self.keys()]
            
            def __iter__(self):
                return iter(self.keys())
            
            def __len__(self):
                return len(self._modules) + len(self._parameters)
            
            def __contains__(self, key):
                return key in self._modules or key in self._parameters

        def dfs_add(dest, keys, value: torch.Tensor):
            if len(keys) == 1:
                dest[keys[0]] = value
                return

            if keys[0] not in dest:
                dest[keys[0]] = DynamicModule()
            dfs_add(dest[keys[0]], keys[1:], value)

        def load_dict(state_dict, prefix):
            out_dict = DynamicModule()
            for key, value in state_dict.items():
                value: torch.Tensor
                if key.startswith(prefix):
                    param_keys = key[len(prefix):].split('.')[1:]
                    dfs_add(out_dict, param_keys, value.clone())
            return out_dict

        self.params_dict = load_dict(state_dict, prefix + 'params_dict')
        self.params_dict.requires_grad_(False)
        return
