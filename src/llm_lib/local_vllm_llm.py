import argparse
import openai
import os
import logging
import json
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)

RETRY_CODES = [429, 502, 503, 504]
RESULTS_PATH = "/gpfs/scratch/bsc70/hpai/storage/projects/heka/chips-design/cvdp/results/"


class vLLM_Instance:
    # ----------------------------------------
    # - Initiate the Model
    # ----------------------------------------

    def __init__(
        self,
        context: str = "You are a helpful assistant.",
        local_ip=None,
        local_port=None,
        model=None,
        dataset=None,
    ):
        self.context = context
        self.local_ip = local_ip
        self.local_port = local_port
        self.model = model
        self.dataset = dataset
        self.file_path = "prompts.jsonl"
        self.debug = False
        self.prompts = []

        self.store_path = os.path.join(RESULTS_PATH, self.dataset, self.model)
        os.makedirs(self.store_path, exist_ok=True)
        self.responses_path = os.path.join(self.store_path, "responses.jsonl")

        api_key = "EMPTY"

        if self.local_ip is None or self.local_port is None:
            raise ValueError("Unable to crete an OpenAI API client without a local IP or PORT")
        elif self.file_path is None or (not os.path.exists(self.file_path)):
            raise ValueError("Unable to read prompt responses file. No input for inference.")

        if self.model is None:
            raise ValueError("No model to query.")

        api_url = f"http://{self.local_ip}:{self.local_port}/v1"
        self.chat = openai.OpenAI(
            api_key=api_key,
            base_url=api_url,
            timeout=99999,
        )
        logging.info(
            f"Created vLLM OpenAI API client using the provided URL ({api_url}). Using model: {self.model}"
        )

    def import_prompts(self) -> None:
        """Import the `prompts.jsonl` file. Responses will be generated on top of that (?)"""
        with open(self.file_path, "r") as fd:
            for line in fd:
                self.prompts.append(json.loads(line))
        logging.info(f"Imported prompts from {self.file_path} for a total of {len(self.prompts)} prompts.")

    def batched_inference(self) -> None:
        """Query the model with a given batch size (batch_outputs) and store the responses"""
        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        messages = []

        # Apply chat templates
        for record in tqdm(self.prompts, desc="Applying chat templates..."):
            conversation = [{"role": "user", "content": record["prompt"]}]
            formatted_prompt = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            messages.append(formatted_prompt)

        # Batch all requests through `completions`
        try:
            # Flush!
            resp = self.chat.completions.create(
                model=self.model,
                prompt=messages,
                n=1,
                max_tokens=16384,
            )

            if len(resp.choices) != len(self.prompts):
                logging.warning(
                    f"Mismatch. Got {len(resp.choices)} responses for {len(self.prompts)} prompts"
                )

            for choice in tqdm(resp.choices, desc="Processing responses..."):
                prompt_index = choice.index
                completion_text = choice.text
                self.prompts[prompt_index]["completion"] = completion_text

        except Exception as e:
            logging.error("Batched completions failed: %s", e)

    def store_responses(self):
        """Store the responses JSONL file on the results dir"""
        logging.info(f"Storing {len(self.prompts)} responses to {self.responses_path}")
        with open(self.responses_path, "w") as f:
            for record in self.prompts:
                if "completion" in record:
                    output_record = {"id": record["id"], "completion": record["completion"]}
                    f.write(json.dumps(output_record) + "\n")


def main():
    # Initialize vLLM client
    parser = argparse.ArgumentParser("vLLM params for local inference")
    parser.add_argument("--local_ip", help="IP of the vLLM inference server.", type=str, required=True)
    parser.add_argument("--local_port", help="Port of the vLLM inference server.", type=str, required=True)
    parser.add_argument("--model", type=str, help="HuggingFace hub name or local path", required=True)
    parser.add_argument("--dataset", type=str, help="Name of the CVDP dataset.", required=True)
    args = parser.parse_args()

    vllm_client = vLLM_Instance(
        local_ip=args.local_ip,
        local_port=args.local_port,
        model=args.model,
    )
    vllm_client.import_prompts()
    vllm_client.batched_inference()
    vllm_client.store_responses()

    # TODO(cristian): verify that model is up (ping it?)
    # TODO(cristian): make sure to let vLLM v1 pick up the model params (it will choose as default the best ones)
    # TODO(cristian): store this in a permanent dir on gpfs


if __name__ == "__main__":
    main()
