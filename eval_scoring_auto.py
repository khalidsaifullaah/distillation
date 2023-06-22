import json
import argparse
import random
import openai
from concurrent.futures import ThreadPoolExecutor
import time
import os

openai.api_key = os.environ['OPENAI_API_KEY']

def get_gpt_response(response):
    instruction = response['instruction']
    response1 = response['response1']
    response2 = response['response2']
    sleep_time = response['sleep_time']
    time.sleep(sleep_time)
    flip = random.choice([True, False])
    if flip:
        response1, response2 = response2, response1
    question = f"""Given the instruction below, which response is better? Please only answer either RESPONSE1 or RESPONSE2: 
        ========================
        ##INSTRUCTION: \n{instruction}
        ========================
        RESPONSE1: \n {response1} 
        ========================
        RESPONSE2: \n {response2} 
        ========================
    """
    success = False
    retry = 0
    while not success and retry <5:
        try:
            gptresponse = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": question}
                    ]
                )['choices'][0]['message']['content']
            print(f"successfully getting a response, {gptresponse}")
            success = True
        except Exception as e:
            print('error:', e)
            print('retry:', retry)
            gptresponse = "FAILED"
            retry += 1
            time.sleep(2)

    return gptresponse, flip

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    file_path = "data_instruct/eval_responses.json"
    with open(file_path, 'r') as file:
        responses = json.load(file)
    # add more sleep time for additional request
    for i, response in enumerate(responses):
        response['sleep_time'] = i*0.02
    model1_score = 0
    model2_score = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        gpt_responses = list(executor.map(get_gpt_response, responses))

    for gptresponse, flip in gpt_responses:
        if (gptresponse == "RESPONSE1" and not flip) or (gptresponse == "RESPONSE2" and flip):
            model1_score += 1
        elif (gptresponse == "RESPONSE2" and not flip) or (gptresponse == "RESPONSE1" and flip):
            model2_score += 1
        else:
            print("Invalid output from gpt4:", gptresponse)

    print("Model1 score:", model1_score)
    print("Model2 score:", model2_score)