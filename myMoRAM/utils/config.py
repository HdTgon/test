from yacs.config import CfgNode as CN

_C = CN()

_C.dataset = ""  # Dataset name
_C.dataset_sequence = []
_C.root = ""  # Directory where datasets are stored
_C.num_shots = -1

_C.backbone = ""
_C.resolution = 224

_C.output_dir = None  # Directory to save the output files (like log.txt and model weights)
_C.model_dir = None
_C.print_freq = 10  # How often (batch) to print training information

_C.seed = None  # use manual seed
_C.deterministic = False  # output deterministic results
_C.gpu = None  # assign a single gpu 
_C.num_workers = 20
_C.prec = "amp"  # fp16, fp32, amp

_C.num_epochs = 10
_C.batch_size = 128
_C.lr = 0.01
_C.weight_decay = 5e-4
_C.loss_type = "CE"
_C.theta = 1.0
_C.lada_k = 16      # lambda_1 in paper
_C.prototype_k = 1  # lambda_2 in paper
_C.image_prototypes_weight_coef = 1.0

_C.t_full_tuning = False  # full fine-tuning
_C.t_bias_tuning = False  # only fine-tuning the bias 
_C.t_ln_tuning = False  # only fine-tuning the layer norm
_C.t_adapter = False
_C.t_adaptformer = False
_C.t_lora = False
_C.t_lora_mlp = False
_C.t_ssf_attn = False
_C.t_ssf_mlp = False
_C.t_ssf_ln = False
_C.t_mask = False   # fine-tuning a specific proportion of all parameters
_C.t_partial = None  # fine-tuning (or parameter-efficient fine-tuning) partial block layers
_C.t_adapter_dim = None  # bottle dimension for adapter / adaptformer / lora.
_C.t_mask_ratio = None
_C.t_mask_seed = None
_C.alpha = 1.0
_C.beta = 1.0

_C.zero_shot = False  # zero-shot CLIP
_C.continue_train = False
_C.continue_train_first = False

# MoRAM-only model/training options. Main/data/logging retain LADA conventions.
_C.checkpoint_path = ""
_C.resume = ""  # optional explicit checkpoint; otherwise model_dir/checkpoint.pth.tar
_C.device = "auto"
_C.template = "a photo of a {}."
_C.iterations = 500
_C.wd = 0.0
_C.ls = 0.0
_C.warmup_length = 0
_C.target_encoder = "all"
_C.target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "ffn_in", "ffn_out"]
_C.rank = 16
_C.topk = 16
_C.temp = 0.01
_C.prune_thre = 0.2
# Use MoRAM recipe, not LADA's AMP/epoch/weight_decay training recipe.
_C.prec = "fp32"
_C.lr = 0.0005
_C.batch_size = 64
