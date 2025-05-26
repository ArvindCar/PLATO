from openai import OpenAI
import os

def preprocessing_string_tool(text):
    for line in text.split('\n'):
        if "[Most similar tool from object database]" in line:
            a =  line[41:]
        if "[Most similar task from task database]" in line:
            b = line[40:-1]
    
    return a, b

def objects_database(database_path):
    items = os.listdir(database_path)
    # print(items)
    return items

def tool_checker_prompt(tool_of_interest, task, database_path):
    objs_database = objects_database(database_path)
    objs_database = ", ".join(objs_database)
    # print(objs_database)
    prompt =  f"""[Task]
You will be given a specific tool and a task. Your objective is to look up objects in the objects database and identify the tool that is most similar to the given tool, \
considering the specified task. The similarity should be based primarily on how the tool is grasped and used. If the given tool is found in the object database, it should be returned directly. \
If it is not found, return the tool that is grasped and used most similarly. You will also be given a task database. After you finish identifying the tool, you need to map the task given with that in the task database and tell \
what is the closest task

Database consists of {objs_database}
Task database consists of 'move from one place to another', 'pick-up', 'handover', 'filing', 'loosening', \
'unscrewing', 'frying'.

[Example]
User Message:
Given Tool: Fork
Task: Eat mac and cheese

Answer:
[Most similar tool from object database] Spoon
[Most similar task from task database] 'pick-up'
[Explanation]: The fork is not listed in the database. The way we hold a spoon for eating mac and cheese is similar to how we hold a fork for eating.

User Message:
Given Tool: {tool_of_interest}
Task: {task}

Answer:
"""
    return prompt

def inferencing_gpt(prompt):
    client = OpenAI()
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {'role': 'user',
        'content': prompt},
      ]
    )
    chatbot_response = response.choices[0].message.content
    # print(chatbot_response, "\n=====\n")
    return preprocessing_string_tool(chatbot_response)

if __name__ == '__main__':
    prompt = tool_checker_prompt('Fork', 'Eating', '/path/to/os_tog/data/OSTOG_physical_experiments')
    similar_tool, similar_task = inferencing_gpt(prompt=prompt)
    print(similar_tool, similar_task)
