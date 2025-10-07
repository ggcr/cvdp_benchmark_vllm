import argparse
import openai
import os
import logging
import json
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm
import asyncio

logging.basicConfig(level=logging.INFO)

RETRY_CODES = [429, 502, 503, 504]
RESULTS_PATH = "/gpfs/scratch/bsc70/hpai/storage/projects/heka/chips-design/cvdp/results/async/"

# TODO(cristian): handle the reasoning CoT & only store the completion


class vLLM_Instance:
    # ----------------------------------------
    # - Initiate the Model
    # ----------------------------------------

    def __init__(
        self,
        local_ip: str,
        local_port: str,
        model: str,
        dataset: str,
        context: str = "You are a helpful assistant.",
    ):
        self.context = context
        self.local_ip = local_ip
        self.local_port = local_port
        self.model = model
        self.dataset = dataset
        self.file_path = "prompts.jsonl"
        self.debug = False
        self.prompts = []

        self.store_path = os.path.join(RESULTS_PATH, self.dataset, self.model.split("/")[-1].replace("/", ""))
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
        # completions api
        self.chat = openai.OpenAI(
            api_key=api_key,
            base_url=api_url,
            timeout=99999,
        )
        # chat api
        self.async_chat = openai.AsyncOpenAI(
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

    def parse_prompt_into_conversation(self, full_prompt: str, problem_id: str = "") -> list:
        """
        One difference between local vs API inference is that in local mode, the system prompt is mixed
        with the user prompt. One handling this without modifying the original codebase is to be naive
        and parse for the system prompt... {system_prompt}\n\n\n{user_prompt}
        """
        if "\n\n\n" in full_prompt:
            parts = full_prompt.split("\n\n\n", 1)
        elif "\n\n" in full_prompt:
            parts = full_prompt.split("\n\n", 1)
        else:
            logging.warning(
                f"[WARNING] Could not find separator for problem {problem_id}. Using entire prompt as user message."
            )
            return [{"role": "user", "content": full_prompt}]

        if len(parts) != 2:
            logging.warning(
                f"[WARNING] Could not split prompt for problem {problem_id}. Using entire prompt as user message."
            )
            return [{"role": "user", "content": full_prompt}]

        system_prompt = parts[0].strip()
        user_prompt = parts[1].strip()

        if not system_prompt:
            logging.warning(f"[WARNING] Empty system prompt for problem {problem_id}")
        if not user_prompt:
            logging.warning(f"[WARNING] Empty user prompt for problem {problem_id}")

        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    async def process_single_prompt(self, record: dict) -> tuple[dict, str]:
        conversation = self.parse_prompt_into_conversation(
            record["prompt"], problem_id=record.get("id", "unknown")
        )

        try:
            resp = await self.async_chat.chat.completions.create(
                model=self.model,
                messages=conversation,
                n=1,
                max_tokens=8192,
            )
            completion_text = resp.choices[0].message.content
            return record, completion_text
        except Exception as e:
            logging.error(f"Chat completion failed for {record.get('id', 'unknown')}: %s", e)
            return record, ""

    async def async_chat_inference(self) -> None:
        """
        Async version of chat_inference that processes all prompts concurrently.
        """
        logging.info(f"Processing {len(self.prompts)} prompts with async chat API...")

        # Create all tasks
        tasks = [self.process_single_prompt(record) for record in self.prompts]

        results = []
        for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Chat completions"):
            result = await coro
            results.append(result)

        for record, completion in results:
            record["completion"] = completion

    def batched_inference(self) -> None:
        """Query the model with a given batch size (batch_outputs) and store the responses"""
        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        messages = []

        # Apply chat templates
        for record in tqdm(self.prompts, desc="Applying chat templates..."):
            conversation = self.parse_prompt_into_conversation(
                record["prompt"], problem_id=record.get("id", "unknown")
            )
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
                n=1,  # I assume that the local_export of prompts already produces N prompts for each p_id
                max_tokens=8192,
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
        local_ip=args.local_ip, local_port=args.local_port, model=args.model, dataset=args.dataset
    )
    vllm_client.import_prompts()

    # Completions API infernece (batched)
    # vllm_client.batched_inference()

    # Chat API inference (serquential)
    # vllm_client.chat_inference()

    # Chat API inference (async)
    asyncio.run(vllm_client.async_chat_inference())

    vllm_client.store_responses()


if __name__ == "__main__":
    main()
