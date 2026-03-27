import os
import torch
import random
from transformers import AutoProcessor
import json
import csv
from tqdm import tqdm
import pandas as pd
import openai
from openai import OpenAI
from utils import encode_image, encode_video, extract_option, count_csv_rows, sort_files_by_number_in_name, compress_video, compress_image, videolist2imglist


class BaselineModel:
    def __init__(self):
        pass

    def test_qa(self):
        raise NotImplementedError

    def cal_acc(self, input_file, output_file):
        with open(output_file, mode='r', encoding='utf-8') as f:
            output_data = json.load(f)

        with open(input_file, mode='r', encoding='utf-8') as f:
            input_data = json.load(f)

        if len(input_data) != len(output_data):
            raise ValueError(f"The number of rows in the output file ({len(output_data)}) does not match the number of items in the gt file ({len(input_data)}).")

        correct_count = 0
        print(output_data)
        for i in range(len(output_data)):
            output_answer = output_data[i]['Answer']
            gt_answer = input_data[i]['closeA']

            if output_answer == gt_answer:
                correct_count += 1
            else:
                pass

        accuracy = correct_count / len(output_data) * 100
        return accuracy


class OpenRouterModel(BaselineModel):
    def __init__(self, api_key=None, model='google/gemini-3-flash-preview'):
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY", None)
        if api_key is None:
            raise ValueError("The api_key is not set")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        self.instruction = f"""
        You are a football expert. You are provided with a question 'Q' and four options 'O1', 'O2', 'O3', and 'O4'.
        Please answer the question with one option that best matches the question (replay with 'O1', 'O2', 'O3', or 'O4').
        Do not include any other text or explanations.
        """
        print(f'Agent uses the model <{model}>')
        self.model = model

    def chat_img(self, input_text, image_path, max_tokens=512):
        try:
            base64_images = []
            for image in image_path:
                base64_image = encode_image(image)
                base64_images.append(base64_image)

            messages = [
                {"role": "system", "content": self.instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_text},
                    ]
                }
            ]
            for image in base64_images:
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image}"
                    }
                })
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content
        except openai.APIStatusError as e:
            if "413" in str(e):
                print("Image too large, compressing and retrying...")
                compressed_images = []
                i = 0
                for image in image_path:
                    i += 1
                    compressed_image_path = compress_image(image, f"tmp_file/compressed_image{i}.jpg", quality=50)
                    compressed_images.append(compressed_image_path)
                return self.chat_img(input_text, compressed_images, max_tokens)
            else:
                raise e

    def chat_video(self, input_text, video_path, max_tokens=512):
        try:
            base64_videos = []
            base64_videos = videolist2imglist(video_path, 25)

            messages = [
                {"role": "system", "content": self.instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_text},
                    ]
                }
            ]
            i = 0
            for video in base64_videos:
                intro = f"These images are uniformly captured from the {i}th video in chronological order. There are a total of {len(video)} pictures."
                messages[1]["content"].append({
                    "type": "text",
                    "text": intro
                })
                for image in video:
                    messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image}"
                        }
                    })
                i += 1
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content
        except openai.APIStatusError as e:
            if "413" in str(e):
                print("video too large, compressing and retrying...")
                compressed_videos = []
                i = 0
                for video in video_path:
                    i += 1
                    compressed_video_path = compress_video(video, f"tmp_file/compressed_video{i}.mp4")
                    compressed_videos.append(compressed_video_path)
                return self.chat_video(input_text, compressed_videos, max_tokens)
            else:
                raise e

    def test_qa(self, input_file, output_file, materials_folder):
        with open(input_file, "r") as f:
            data = json.load(f)
        answers = []
        for item in tqdm(data):
            id_num = item["id"]
            question = item["Q"]
            options = []
            for i in range (1, 10):
                options.append(item.get(f"O{i}", None))
            option_num = 0
            prompt = f"Q: {question}\n"
            for i in range(1, 10):
                if options[i-1]:
                    prompt += f"O{i}: {options[i-1]}\n"
                    option_num += 1

            old_material_path = item["materials"]
            material_path = []
            try:
                if old_material_path:
                    ## Path Error: Q5
                    if '/q5/' in old_material_path[0]:
                        new_one = [os.path.join(old_material_path[0], p) for p in os.listdir(os.path.join(materials_folder, old_material_path[0])) if p.endswith('.jpg')]
                        old_material_path = new_one
                    # else:
                    #     continue  # only process error q5, delete if process all questions

                    for path in old_material_path:
                        material_path.append(os.path.join(materials_folder, path))
                    if not os.path.isfile(material_path[0]):
                        material_path = sort_files_by_number_in_name(material_path[0])
                    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.heic'}
                    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg'}
                    if os.path.splitext(material_path[0])[1].lower() in image_exts:  # ext (.jpg .mp4)
                        reply = self.chat_img(prompt, material_path)
                    elif os.path.splitext(material_path[0])[1].lower() in video_exts:
                        reply = self.chat_video(prompt, material_path)
                else:
                    reply = self.chat_img(prompt, [])
                answer = extract_option(reply)
            except Exception as e:
                print(f"Error processing item id {id_num}: {e}")
                answer = None

            # if the model does not return a valid option, randomly select one
            if not answer:
                answer = random.choice([f'O{i}' for i in range(1, option_num+1)])
            answers.append({"id": id_num, "Answer": answer})
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)
        return