import os
import pandas as pd
import argparse
from model import OpenRouterModel


def main():
    # Setting up command-line arguments
    parser = argparse.ArgumentParser(description="Run QA tests with different models.")
    parser.add_argument('--input_file', required=True, help='The input JSON file for the QA task')
    parser.add_argument('--output_file', required=True, help='The output JSON file to store the results')
    parser.add_argument('--materials_folder', required=True, help='Folder containing material files')
    parser.add_argument('--model', default='google/gemini-3-flash-preview', help='The model to use')
    parser.add_argument('--api_key', default=None, help='API key for GPT model (if applicable)')
    parser.add_argument('--model_path', default=None, help='Model path for Qwen model (if applicable)')
    args = parser.parse_args()

    agent = OpenRouterModel(api_key=args.api_key, model=args.model)

    # Running the QA test with the given model and file
    agent.test_qa(args.input_file, args.output_file, args.materials_folder)

    # Calculate the accuracy if not challenge.
    print(f"Results saved to {args.output_file}")
    if "challenge" not in args.input_file:
        accuracy = agent.cal_acc(args.input_file, args.output_file)
        print(f"Accuracy of {args.model} model: {accuracy:.2f}%, ")

if __name__ == "__main__":
    main()
