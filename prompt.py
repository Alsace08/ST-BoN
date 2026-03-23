DATASET_PROMPTS = {
    "MATH-500": "Question: {raw_input}\nPlease reason step by step, and put your final answer within \\boxed{{}}\n",
    "AIME2024": "Question: {raw_input}\nPlease reason step by step, and put your final answer within \\boxed{{}}\n",
    "AIME2025": "Question: {raw_input}\nPlease reason step by step, and put your final answer within \\boxed{{}}\n",
    "amc23": "Question: {raw_input}\nPlease reason step by step, and put your final answer within \\boxed{{}}\n",
    "MMLU": '''
Answer the following multiple choice question, and put your final answer within \\boxed{{}} at the last line, using the format \\boxed{{$LETTER}} where $LETTER is one of ABCD.

Question:
{raw_input} 
''',
    "theoremqa": '''
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
Please read a math problem, and then think step by step to derive the answer. The answer is decided by Answer Type.
If the Answer type in [bool], the answer needs to be True or False.
Else if the Answer type in [integer, float] , The answer needs to be in numerical form.
Else if the Answer type in [list of integer, list of float] , the answer needs to be a list of number like [2, 3, 4].
Else if the Answer type in [option], the answer needs to be an option like (a), (b), (c), (d).
You need to output the answer in your final sentence like 'Therefore, the answer is ...'.

Please reason step by step, and put your final answer within \\boxed{{}}.

### Question: 
''' + "{raw_input}\n\n### Answer_type: {answer_type}\n\n### Response:"
}


