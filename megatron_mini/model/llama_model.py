import torch
from megatron_mini import get_args
from megatron_mini.model.module import MegatronModule
from megatron_mini.model.transformer import LLaMATransformer


class LLaMAModel(MegatronModule):
    """Code Generation Model for Multilingual Program Synthesis."""

    def __init__(self, parallel_output=False):
        super(LLaMAModel, self).__init__()
        args = get_args()

        self.parallel_output = parallel_output

        self._language_model_key = "llama_model"
        self.language_model = LLaMATransformer(
            init_method=lambda x:x,
            output_layer_init_method=lambda x:x
        )
        
    def forward(self, tokens: torch.Tensor, start_pos: int, return_hidden=False):

        # Language model.
        lm_logits = self.language_model(tokens, start_pos, return_hidden)

        return lm_logits
    
    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)