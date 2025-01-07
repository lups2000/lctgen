from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os

folder = os.path.dirname(__file__)
print(folder)

from lctgen.core.basic import BasicLLM
from lctgen.core.registry import registry

@registry.register_llm(name='codex')
class CodexModel(BasicLLM):
    def __init__(self, config):
        super().__init__(config)
        self.codex_cfg = config.LLM.CODEX
        prompt_path = os.path.join(folder, 'prompts', self.codex_cfg.PROMPT_FILE)
        self.base_prompt = open(prompt_path).read().strip()

        sys_prompt_file = self.codex_cfg.SYS_PROMPT_FILE
        if sys_prompt_file:
            sys_prompt_path = os.path.join(folder, 'prompts', sys_prompt_file)
            self.sys_prompt = open(sys_prompt_path).read().strip()
        else:
            self.sys_prompt = "Only answer with a function starting def execute_command."

        # Load Hugging Face model and tokenizer with access token
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.codex_cfg.MODEL_NAME,
            use_auth_token=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.codex_cfg.MODEL_NAME,
            use_auth_token=True
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        self.terminators = [
            self.pipe.tokenizer.eos_token_id,
            self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def prepare_prompt(self, query, base_prompt):
        extended_prompt = base_prompt.replace("INSERT_QUERY_HERE", query)
        return extended_prompt

    def llm_query(self, extended_prompt):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(device)
        
        full_prompt = f"{self.sys_prompt}\n\nNow this is the query to process and you should output the result: {extended_prompt}"
        
        outputs = self.pipe(
            full_prompt,
            eos_token_id=self.terminators,
            temperature=self.codex_cfg.TEMPERATURE,
            top_p=1.,
        )
        
        response = outputs[0]["generated_text"]
        return response

    def post_process(self, response):
        return response
  
@registry.register_llm(name='null')
class NullCodex(CodexModel):
  def __init__(self, config):
    super().__init__(config)
  
  def llm_query(self, extended_prompt):
    # for debugging
    return extended_prompt
  