"""This file is used to score the models on the eval set. 
It will print out the instruction, response1, and response2, and 
ask the user which response they prefer. It will then keep track of the score for each model. 
The responses are shuffled before being presented to the user. 
"""
import json
import argparse
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="data_instruct/eval_responses.json")
    args = parser.parse_args()

    file_path = args.file_path
    with open(file_path, 'r') as file:
        responses = json.load(file)
    # shuffle responses
    random.shuffle(responses)
    model1_score = 0
    model2_score = 0
    for response in responses:
        instruction = response['instruction']
        response1 = response['response1']
        response2 = response['response2']
        
        print(f"INSTRUCTION: \n {instruction}")
        print("=====================================")
        print(f"RESPONSE1: \n {response1}")
        print("=====================================")
        print(f"RESPONSE2: \n {response2}")
        print("=====================================")

        better_response = input("Which response do you prefer? (1 or 2) \n")
        
        flip = random.choice([True, False])
        if (better_response == "1" and not flip) or (better_response == "2" and flip):
            model1_score += 1
        elif (better_response == "2" and not flip) or (better_response == "1" and flip):
            model2_score += 1
        else:
            print("Invalid input")
        print(f"current score: model1: {model1_score}, model2: {model2_score}")
    


