import sys, json, argparse, datetime
sys.path.append('/home/zhaosiyao/SoccerAgent')
import os
import warnings
warnings.filterwarnings("ignore")

class Logger(object):
    def __init__(self, filename="output.log"):
        self.terminal = sys.__stdout__
        # Try to use unbuffered mode if available
        if hasattr(self.terminal, 'buffer'):
            self.terminal = os.fdopen(self.terminal.fileno(), 'w', buffering=1)
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "/home/zhaosiyao/SoccerAgent/log"
os.makedirs(log_dir, exist_ok=True)
output_file_arg = None
for i, arg in enumerate(sys.argv):
    if arg == "--output_file" and i + 1 < len(sys.argv):
        output_file_arg = sys.argv[i+1]
        break
if output_file_arg:
    base_name = os.path.basename(output_file_arg).replace(".json", ".log")
    log_path = os.path.join(log_dir, base_name)
else:
    log_path = os.path.join(log_dir, f"running_log_{timestamp}.log")
sys.stdout = Logger(log_path)
sys.stderr = sys.stdout

import os
import argparse
# Force qwen-vl-utils to use torchvision as the video reader backend to avoid decord/torchcodec errors
os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"

from multiagent_platform import EXECUTE_TOOL_CHAIN
from openai import OpenAI
from toolbox.utils.all_devices import API_MODEL

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_BASE_URL'))

INSTRUCTION = f"""
You are a football expert. You are provided with a question 'Q' and four options 'O1', 'O2', 'O3', and 'O4'.
Before I have used a helpful soccer multi-agent system to solve this process, I will tell you the total process of how agent deal with this problem.
Please answer the question with one option that best matches the question (replay with 'O1', 'O2', 'O3', or 'O4').
Do not include any other text or explanations!!!
"""

def workflow(input_text, Instruction=INSTRUCTION, follow_up_prompt=None, max_tokens_followup=1500):
    completion = client.chat.completions.create(
        model=API_MODEL,
        messages=[
            {"role": "system", "content": Instruction},
            {"role": "user", "content": input_text}
        ],
    )
    first_round_reply = completion.choices[0].message.content
    return first_round_reply

import re

def process_football_question(input_dict, materials_folder):
    if "openA_process" in input_dict and "answer" in input_dict:
        return input_dict

    question = input_dict.get("Q", "")
    old_material_path = input_dict.get("materials", "")
    materials = []
    if old_material_path:
        # Path Error: Q5
        if '/q5/' in old_material_path[0]:
            new_one = [os.path.join(old_material_path[0], p) for p in os.listdir(os.path.join(materials_folder, old_material_path[0])) if p.endswith('.jpg')]
            old_material_path = new_one
        materials = [os.path.join(materials_folder, material) for material in old_material_path]

    options = {key: value for key, value in input_dict.items() if key.startswith("O")}
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])

    openA_process = EXECUTE_TOOL_CHAIN(question, materials, options_str)

    prompt = f"""
This football question is "{question}". The four corresponding options are:
{options_str}

The processing through the multi-agent platform is as follows:
{openA_process}

Please provide your answer:
"""

    processed_prompt = workflow(prompt)

    answer_match = re.search(r"O\d+", processed_prompt)
    answer = answer_match.group(0) if answer_match else None

    # Extract all errors from the multi-agent process
    errors = []
    if openA_process:
        # We split the string by <Tool>...</Tool> to associate each tool with its following Answer
        parts = re.split(r"<Tool>(.*?)</Tool>", openA_process)

        # parts[0] is text before first <Tool>
        # parts[1] is Tool 1
        # parts[2] is text after Tool 1 (containing <Answer>...</Answer>)
        # parts[3] is Tool 2
        # ...

        for i in range(1, len(parts), 2):
            tool_name = parts[i].strip()
            following_text = parts[i+1]

            # Find the Answer block right after this tool
            answer_match = re.search(r"<Answer>\s*(Error occurred:.*?|Failed:.*?|Failed to read JSON file:.*?|Unsupported file extension:.*?|cannot identify image file.*?)\s*</Answer>", following_text, re.DOTALL)
            if answer_match:
                error_msg = answer_match.group(1).strip()
                # Create a dictionary for this error and add it to the list if not already present
                error_dict = {tool_name: error_msg}
                if error_dict not in errors:
                    errors.append(error_dict)

    result_dict = input_dict.copy()
    result_dict["openA_process"] = openA_process
    result_dict["Answer"] = answer
    result_dict["Error"] = errors if errors else None

    return result_dict


from tqdm import tqdm
import json

def process_json_file(input_file, output_file, materials_folder):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        progress_bar = tqdm(data_list, desc="Processing (Accuracy: N/A)", unit="item")

        for i, item in enumerate(progress_bar):
            print(f'Processing problem {i}: {item["Q"]}')
            try:
                updated_item = process_football_question(item, materials_folder)
                data_list[i] = updated_item

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data_list, f, ensure_ascii=False, indent=4)

            except ValueError as ve:
                print(f"ValueError processing item {i}: {ve}")
                continue

            except Exception as e:
                print(f"Unexpected error processing item {i}: {e}")
                continue

            if 'challenge' not in input_file:
                correct_count = 0
                total_count = 0
                for entry in data_list:
                    if "openA_process" in entry and "Answer" in entry:
                        total_count += 1
                        if entry["Answer"] == entry.get("closeA"):
                            correct_count += 1

                accuracy = correct_count / total_count if total_count > 0 else 0

                progress_bar.set_description(f"Processing (Accuracy: {accuracy:.2%})")
                progress_bar.refresh()

        print(f"Processing completed. Output saved to {output_file}")

    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSON file containing football questions.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSON file. See https://huggingface.co/datasets/Homie0609/SoccerBench/raw/main/qa/q1.json as an example")
    parser.add_argument("--output_file", type=str, help="Path to save the output JSON file. You can just set an json path.")
    parser.add_argument('--materials_folder', type=str, help='Folder containing material files')

    args = parser.parse_args()
    process_json_file(args.input_file, args.output_file, args.materials_folder)
