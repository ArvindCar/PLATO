def ProcessString(input_string):
    input_string = input_string.lower()
    steps_list_part = input_string.split("steps list:")[1].strip()
    steps = [step.split('. ', 1)[1] for step in steps_list_part.split('\n')]
    steps_nested = []
    for step in steps:
        first_split = step.split(": ", 1)
        second_split = first_split[1].split(" + ", 1)
        nested_step = [first_split[0]] + second_split
        steps_nested.append(nested_step)
    print(steps_nested)
    # return steps_nested

strins = '''Since the gripper doesn't need to perform any Go-to or Tilt operations, just the Grasp step is sufficient.

Steps List:
1. Grasp: 0'''
ProcessString(strins)
