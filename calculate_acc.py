import json
import argparse


def cal_acc(input_file, output_file):
    with open(output_file, mode='r', encoding='utf-8') as f:
        output_data = json.load(f)

    with open(input_file, mode='r', encoding='utf-8') as f:
        input_data = json.load(f)

    if len(input_data) != len(output_data):
        raise ValueError(f"The number of rows in the output file ({len(output_data)}) does not match the number of items in the gt file ({len(input_data)}).")

    correct_count = 0
    total_count = 0
    for i in range(len(output_data)):
        if 'Answer' in output_data[i]:
            total_count += 1
            output_answer = output_data[i]['Answer']
            gt_answer = input_data[i]['closeA']
            if output_answer == gt_answer:
                correct_count += 1
            else:
                pass

    accuracy = correct_count / total_count * 100
    return accuracy, correct_count, total_count

def main():
    # Setting up command-line arguments
    parser = argparse.ArgumentParser(description="Run QA tests with different models.")
    parser.add_argument('--input_file', required=True, help='The input JSON file for the QA task')
    parser.add_argument('--output_file', required=True, help='The output JSON file to store the results')
    args = parser.parse_args()

    accuracy, correct_count, total_count = cal_acc(args.input_file, args.output_file)
    print(f"Result file: {args.output_file}")
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")

if __name__ == "__main__":
    main()
