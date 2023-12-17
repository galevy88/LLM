from torch import cuda, bfloat16
import transformers

class LlamaModel:
    _instance = None

    def __new__(cls, hf_auth, model_id):
        if cls._instance is None:
            cls._instance = super(LlamaModel, cls).__new__(cls)
            cls._instance._init_model(hf_auth, model_id)
        return cls._instance

    def _init_model(self, hf_auth, model_id):
        
        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        print(device)

        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16,
        )

        
        model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
        )

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=hf_auth
        )


        self.model.eval()

    def get_model(self):
        return self.model
