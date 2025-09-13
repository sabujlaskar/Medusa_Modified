# Batch, non-interactive CLI for Medusa
# Usage examples:
#   python -m medusa.inference.batch_cli --model mistralai/Mistral-7B-v0.1 \
#       --input-file prompts.txt --output-file outputs.jsonl --max-steps 256
#
#   python -m medusa.inference.batch_cli --model <path_or_hf_id> \
#       --input-file prompts.jsonl --jsonl

import argparse, sys, os, json
import torch
from typing import Iterable, Union

from fastchat.model.model_adapter import get_conversation_template
from fastchat.conversation import get_conv_template
from medusa.model.medusa_model import MedusaModel

def _read_prompts_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            # expect {"prompt": "...", "id": "..."}; id optional
            yield {"id": obj.get("id"), "prompt": obj["prompt"]}

def _consume_outputs(outputs: Union[str, Iterable[str]]) -> str:
    """Medusa's medusa_generate may yield text chunks. Join them safely."""
    if isinstance(outputs, str):
        return outputs
    try:
        return "".join(list(outputs))
    except TypeError:
        # If it's not an iterable of strings, just convert to str
        return str(outputs)

def main():
    ap = argparse.ArgumentParser("Medusa batch generator (non-interactive)")
    parser.add_argument("--model", type=str, required=True, help="Model name or path.")
    ap.add_argument("--input-file", required=True, help="Text file or JSONL of prompts")
    ap.add_argument("--output-file", default=None, help="Write JSONL with fields: id, prompt, output")
    parser.add_argument("--load-in-8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit quantization")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-steps", type=int, default=512, help="Max generation steps (tokens) for Medusa")
    ap.add_argument("--conv-template", type=str, default=None, help="Override FastChat conversation template name")
    ap.add_argument("--conv-system-msg", type=str, default=None, help="Override system message")
    ap.add_argument("--print-separator", default="-"*60, help="Separator line printed between outputs")
    args = ap.parse_args()

    # Load model
    model = MedusaModel.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    tokenizer = model.get_tokenizer()
    dev = model.base_model.device

    # Prepare reader
    prompts_iter = _read_prompts_jsonl(args.input_file)

    # Prepare writer
    writer = None
    if args.output_file:
        writer = open(args.output_file, "w", encoding="utf-8")

    # Optional custom template
    def build_conv():
        if args.conv_template:
            conv = get_conv_template(args.conv_template)
        else:
            conv = get_conversation_template(args.model)
        if args.conv_system_msg:
            conv.set_system_message(args.conv_system_msg)
        return conv

    torch.set_grad_enabled(False)

    for idx, item in enumerate(prompts_iter, start=1):
        pid = item["id"] if item["id"] is not None else idx
        user_prompt = item["prompt"]

        # Build a fresh conversation for each prompt
        conv = build_conv()
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        enc = tokenizer(full_prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(dev)

        # Run Medusa generation (sequential, non-streaming)
        try:
            out_chunks = model.medusa_generate(
                input_ids,
                temperature=args.temperature,
                max_steps=args.max_steps,
            )
            text = _consume_outputs(out_chunks).strip()
        except KeyboardInterrupt:
            print("\n[Interrupted] stopping at item:", pid, file=sys.stderr)
            break

        # Print to stdout (sequential)
        print(args.print_separator)
        print(f"[{pid}] Prompt:\n{user_prompt}\n")
        print(f"[{pid}] Output:\n{text}\n")

        # Save JSONL if requested
        if writer:
            writer.write(json.dumps({"id": pid, "prompt": user_prompt, "output": text}) + "\n")
            writer.flush()

    if writer:
        writer.close()

if __name__ == "__main__":
    main()
