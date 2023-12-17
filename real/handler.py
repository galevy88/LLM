from model_singleton import LlamaModel
import transformers
from langchain.llms import HuggingFacePipeline
import time
import warnings
warnings.filterwarnings('ignore')



hf_auth = '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'
model_id = 'meta-llama/Llama-2-13b-hf'


def get_tokenizer():
    return transformers.AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)

def get_hf_pipline():
    return transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
    )

model_instance = LlamaModel(hf_auth, model_id)
model = model_instance.get_model()
tokenizer = get_tokenizer()
hf_pipline = get_hf_pipline()


llm = HuggingFacePipeline(pipeline=hf_pipline)

p ="What is Anti Matter?"

print("START WORK ON TEXT")

start_time = time.time()  # Start timing

ans = llm(prompt=p)

end_time = time.time()  # End timing

print(ans)
print(f"Time taken: {end_time - start_time} seconds")

