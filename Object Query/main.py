
import pandas as pd
import matplotlib.pyplot as plt
from object_query import ObjectQuery
from chatgpt_interface import ChatGPTInterface

import numpy as np

import numpy as np

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import json
from object_query import ObjectQuery
from chatgpt_interface import ChatGPTInterface

def main():
    chatgpt_interface = ChatGPTInterface()
    df = pd.read_csv("group_perspective_objects_and_severities_GT.csv")
    results = []

    for index, row in df.iterrows():
        object_name = row["Items"]
        ground_truth_severity = row["Severity"]

        object_query = ObjectQuery(object_name)
        prompt = object_query.generate_prompt()

        # Single call to ChatGPT
        GPT_response = chatgpt_interface.get_gpt_response(prompt)
        if GPT_response is not None:
            severity_level = ObjectQuery.get_Severity(GPT_response.strip().lower())
        else:
            GPT_response = "unknown"
            severity_level = ObjectQuery.get_Severity(GPT_response)

        results.append({
            "object_name": object_name,
            "ground_truth_severity": ground_truth_severity,
            "chatgpt_response": GPT_response,
            "severity_level": severity_level
        })

        print(f"Object: {object_name}, Ground Truth: {ground_truth_severity}, ChatGPT Response: {GPT_response}, Severity Level: {severity_level}")

    with open("chatgpt_severity_responses.json", "w") as f:
        json.dump(results, f, indent=4)

    df["chatgpt_severity"] = [result["severity_level"] for result in results]

    correct_predictions = (df["Severity"] == df["chatgpt_severity"]).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions * 100

    partial_accuracy_score = 0
    for index, row in df.iterrows():
        ground_truth = row["Severity"]
        chatgpt_prediction = row["chatgpt_severity"]

        if pd.isna(chatgpt_prediction):
            continue

        error = abs(ground_truth - chatgpt_prediction)
        if error == 0:
            partial_accuracy_score += 1
        elif error <= 5:
            partial_accuracy_score += 0.75
        elif error <= 9:
            partial_accuracy_score += 0.0
        else:
            partial_accuracy_score += 0

    partial_accuracy = (partial_accuracy_score / total_predictions) * 100

    print(f"Accuracy of ChatGPT responses (exact matches): {accuracy:.2f}%")
    print(f"Partial Accuracy of ChatGPT responses (with partial credit): {partial_accuracy:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(df["Items"], df["Severity"], label="Subjective Ground Truth", marker='o')
    plt.plot(df["Items"], df["chatgpt_severity"], label="ChatGPT", marker='x')
    plt.xlabel("Items")
    plt.ylabel("Severity Level")
    plt.title("Comparison of Ground Truth and ChatGPT Severity Levels")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

'''
def main():
    chatgpt_interface = ChatGPTInterface()
    df = pd.read_csv("objects_and_severities.csv") 
    chat_severities = []

    severity_numeric_mapping = {
        "high": 10,
        "medium": 5,
        "low": 1,
        "unknown": 0  
    }

    for index, row in df.iterrows():
        object_name = row["Items"]  
        ground_truth_severity = row["Severity"] 

        object_query = ObjectQuery(object_name)
        prompt = object_query.generate_prompt()

        responses = []

        for _ in range(5):  
            GPT_response = chatgpt_interface.get_gpt_response(prompt)
            if GPT_response is not None:
                responses.append(GPT_response)
            else:
                responses.append("unknown")  

        severity_values = []
        for response in responses:
            if response == "high":
                severity_values.append(severity_numeric_mapping["high"])
            elif response == "medium":
                severity_values.append(severity_numeric_mapping["medium"])
            elif response == "low":
                severity_values.append(severity_numeric_mapping["low"])
            else:
                severity_values.append(severity_numeric_mapping["unknown"])

        valid_responses = [value for value in severity_values if value != 0] 

        if valid_responses:  
            avg_severity = np.mean(valid_responses)
            if avg_severity >= 7:
                avg_severity_text = "high"
            elif avg_severity >= 3:
                avg_severity_text = "medium"
            else:
                avg_severity_text = "low"
        else:
            avg_severity_text = "unknown"

        chat_severities.append(avg_severity_text)

        print(f"Object: {object_name}, Ground Truth: {ground_truth_severity}, ChatGPT Average Response: {avg_severity_text}")

    df["chatgpt_severity"] = chat_severities
    df["ground_truth_numeric"] = df["Severity"]
    df["chatgpt_numeric"] = df["chatgpt_severity"].map({
        "high": 10,
        "medium": 5,
        "low": 1,
        "unknown": 0
    })

    correct_predictions = (df["ground_truth_numeric"] == df["chatgpt_numeric"]).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions * 100

    partial_accuracy_score = 0
    for index, row in df.iterrows():
        ground_truth = row["ground_truth_numeric"]
        chatgpt_prediction = row["chatgpt_numeric"]
        error = abs(ground_truth - chatgpt_prediction)
        if error == 0:
            partial_accuracy_score += 1
        elif error <= 5:
            partial_accuracy_score += 0.75
        elif error <= 9:
            partial_accuracy_score += 0.0
        else:
            partial_accuracy_score += 0

    partial_accuracy = (partial_accuracy_score / total_predictions) * 100

    print(f"Accuracy of ChatGPT responses (exact matches): {accuracy:.2f}%")
    print(f"Partial Accuracy of ChatGPT responses (with partial credit): {partial_accuracy:.2f}%")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(df["Items"], df["ground_truth_numeric"], label="Subjective Ground Truth", marker='o')
    plt.plot(df["Items"], df["chatgpt_numeric"], label="ChatGPT", marker='x')
    plt.xlabel("Items")
    plt.ylabel("Severity Level")
    plt.title("Comparison of Ground Truth and ChatGPT Severity Levels")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
'''



'''
from object_query import ObjectQuery
from chatgpt_interface import ChatGPTInterface

def main():
    object_query = ObjectQuery("baby")  # used to set the object label
    chatgpt_interface = ChatGPTInterface()

    prompt = object_query.generate_prompt()
    print("Generated Prompt:", prompt)

    GPT_response = chatgpt_interface.get_gpt_response(prompt)
    if GPT_response is not None:
        print("ChatGPT Response:", GPT_response)
    else:
        print("The response was not as intended. Please check the printed response above.")

    print("severity level: ", ObjectQuery.get_Severity(GPT_response))

if __name__ == "__main__":
    main()
'''

'''
#Old Code From When ChatGPT assigned the values of 
from object_query import ObjectQuery
from chatgpt_interface import ChatGPTInterface
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
file_path = '/Users/kozasound/Desktop/LLM High Level Controler/Object Query/random_items_with_severity_complete.csv'
data = pd.read_csv(file_path)


chatgpt_interface = ChatGPTInterface()
predicted_severity = []

for _, row in data.iterrows():
    item = row["Item"]
    actual_severity = row["Severity"]
    object_query = ObjectQuery(environment_object=item)
    prompt = object_query.generate_prompt()
    print(f"Generated Prompt for {item}: {prompt}")

    response = chatgpt_interface.get_integer_response(prompt)
    if response is None:
        print(f"Non-integer response for {item}. Assigning severity as -1.")
        response = -1

    predicted_severity.append(response)

data["Predicted_Severity"] = predicted_severity

valid_mask = (data["Severity"] != -1) & (data["Predicted_Severity"] != -1)
if valid_mask.sum() > 0:
    accuracy = (data.loc[valid_mask, "Severity"] == data.loc[valid_mask, "Predicted_Severity"]).mean() * 100
else:
    accuracy = None

plt.figure(figsize=(12, 6))
plt.plot(data["Item"], data["Severity"], label="Actual Severity", marker='o')
plt.plot(data["Item"], data["Predicted_Severity"], label="Predicted Severity", marker='x')
plt.xticks(rotation=90)
plt.xlabel("Item")
plt.ylabel("Severity")
plt.title("Actual vs Predicted Severity")
plt.legend()
plt.tight_layout()
plt.show()

if accuracy is not None:
    print(f"Prediction Accuracy (ignoring -1s): {accuracy:.2f}%")
else:
    print("No valid comparisons for accuracy calculation.")
'''

'''
from object_query import ObjectQuery
from chatgpt_interface import ChatGPTInterface

def main():
    object_query = ObjectQuery("baby")  # used to set the object label
    chatgpt_interface = ChatGPTInterface()

    prompt = object_query.generate_prompt()
    print("Generated Prompt:", prompt)

    GPT_response = chatgpt_interface.get_gpt_response(prompt)
    if GPT_response is not None:
        print("ChatGPT Response:", GPT_response)
    else:
        print("The response was not as intended. Please check the printed response above.")

    print("severity level: ", ObjectQuery.get_Severity(GPT_response))

if __name__ == "__main__":
    main()
'''
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from object_query import ObjectQuery
from chatgpt_interface import ChatGPTInterface

def main():
    chatgpt_interface = ChatGPTInterface()
    df = pd.read_csv("objects_and_severities.csv") 
    chat_severities = []


    for index, row in df.iterrows():
        object_name = row["Items"]  
        ground_truth_severity = row["Severity"] 

        object_query = ObjectQuery(object_name)
    
        prompt = object_query.generate_number_prompt() 
        
        responses = []
        for _ in range(5):  
            GPT_response = chatgpt_interface.get_gpt_response(prompt)
            if GPT_response is not None:
                responses.append(GPT_response)
            else:
                responses.append("unknown")  

        severity_values = []
        for response in responses:
            try:
                numeric_value = float(response)

                if 1 <= numeric_value <= 10:
                    severity_values.append(numeric_value)
                else:
                    severity_values.append(None)  
            except ValueError:

                severity_values.append(None)

        valid_responses = [value for value in severity_values if value is not None]  

        if valid_responses:
            avg_severity = np.mean(valid_responses)

            chat_severities.append(avg_severity)
        else:
            chat_severities.append(None)

        print(f"Object: {object_name}, Ground Truth: {ground_truth_severity}, ChatGPT Average Response: {avg_severity if valid_responses else 'unknown'}")

    df["chatgpt_numeric"] = chat_severities 

    severity_mapping = {
        "high": 10,
        "medium": 5,
        "low": 1,
        "unknown": 0
    }

    df["ground_truth_numeric"] = df["Severity"]

    correct_predictions = (df["ground_truth_numeric"] == df["chatgpt_numeric"]).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions * 100

    partial_accuracy_score = 0
    for index, row in df.iterrows():
        ground_truth = row["ground_truth_numeric"]
        chatgpt_prediction = row["chatgpt_numeric"]
        
        if pd.isna(chatgpt_prediction):
            continue  

        error = abs(ground_truth - chatgpt_prediction)
        if error == 0:
            partial_accuracy_score += 1
        elif error <= 5:
            partial_accuracy_score += 0.75
        elif error <= 9:
            partial_accuracy_score += 0.0
        else:
            partial_accuracy_score += 0

    partial_accuracy = (partial_accuracy_score / total_predictions) * 100

    print(f"Accuracy of ChatGPT responses (exact matches): {accuracy:.2f}%")
    print(f"Partial Accuracy of ChatGPT responses (with partial credit): {partial_accuracy:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(df["Items"], df["ground_truth_numeric"], label="Subjective Ground Truth", marker='o')
    plt.plot(df["Items"], df["chatgpt_numeric"], label="ChatGPT", marker='x')
    plt.xlabel("Items")
    plt.ylabel("Severity Level")
    plt.title("Comparison of Ground Truth and ChatGPT Severity Levels")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
'''

'''
#Old Code From When ChatGPT assigned the values of  
from object_query import ObjectQuery
from chatgpt_interface import ChatGPTInterface
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
file_path = '/Users/kozasound/Desktop/LLM High Level Controler/Object Query/random_items_with_severity_complete.csv'
data = pd.read_csv(file_path)


chatgpt_interface = ChatGPTInterface()
predicted_severity = []

for _, row in data.iterrows():
    item = row["Item"]
    actual_severity = row["Severity"]
    object_query = ObjectQuery(environment_object=item)
    prompt = object_query.generate_prompt()
    print(f"Generated Prompt for {item}: {prompt}")

    response = chatgpt_interface.get_integer_response(prompt)
    if response is None:
        print(f"Non-integer response for {item}. Assigning severity as -1.")
        response = -1

    predicted_severity.append(response)

data["Predicted_Severity"] = predicted_severity

valid_mask = (data["Severity"] != -1) & (data["Predicted_Severity"] != -1)
if valid_mask.sum() > 0:
    accuracy = (data.loc[valid_mask, "Severity"] == data.loc[valid_mask, "Predicted_Severity"]).mean() * 100
else:
    accuracy = None

plt.figure(figsize=(12, 6))
plt.plot(data["Item"], data["Severity"], label="Actual Severity", marker='o')
plt.plot(data["Item"], data["Predicted_Severity"], label="Predicted Severity", marker='x')
plt.xticks(rotation=90)
plt.xlabel("Item")
plt.ylabel("Severity")
plt.title("Actual vs Predicted Severity")
plt.legend()
plt.tight_layout()
plt.show()

if accuracy is not None:
    print(f"Prediction Accuracy (ignoring -1s): {accuracy:.2f}%")
else:
    print("No valid comparisons for accuracy calculation.")
'''