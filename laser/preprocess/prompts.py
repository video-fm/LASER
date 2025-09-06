locations = ["early", "mid", "late"]
durations = ["long", "medium", "short"]

user = '''
You are a super user in logic programming. You are also an expert at structured data extraction.
'''

context = f'''
You will describe the event length and location in both natural language and fraction of the video. 
The natural language description of the locations in the video can be: {', '.join(locations)}.
The natural language description of the durations of the event can be: {', '.join(durations)}
Examples of precise video locations: [1/4, 1/2], [2/3, 1].
Examples of event durations: 1/4, 2/3, 1.
'''

example1 = '''
Caption: A man carries a child and walks to the left from behind a woman holding another child.
Video_ID: "0be30efe",
Action json: {"video_id": "0be30efe", "sequential descriptions": [ "man A carry child B, women C hold child D, man A is behind women C", "man A walk", "man A at left" ], "time stamps": { "1": { "description": [ "man A carry child B", "women C hold child D", "man A is behind women C" ], "programmatic": [ "carrying(A, B)", "name(A, man)", "name(B, child)", "holding(C, D)", "name(C, women)", "name(D, man)", "behind(A, C)",  ], "duration": "short", "duration precise": "1/4", "video location": "early",  "video location precise": "[0, 1/4]" }, "2": { "description": [ "man A walk" ], "programmatic": [ "walk(A)", ], "duration": "medium", "duration precise": "1/2", "video location": "mid", "video location precise": "[1/4, 3/4]" }, "3": { "description": [ "man A at left" ], "programmatic": [ "left(A)" ], "duration": "short", "duration precise": "1/4", "video location": "late", "video location precise": "[3/4, 1]"}}}
'''

example2 = '''
Caption: The woman rocks and holds the child, singing a birthday song together with another woman to celebrate the birthday of the girl.
video_id: "P01_03",
Action json: {"video_id": "P01_03", "sequential descriptions": [ "woman A rocks and holds the child B, woman A and women C sings birthday song" ], "time stamps": { "1": { "decription": [ "woman A rocks and holds the child B, woman A and women C sings birthday song" ], "programmatic": [ "rock(A, B)", "hold(A, B)", "sing(A)", "sing(B)" ], "duration": "long", "duration precise": "1", "video location": "mid",  "video location precise": "[0, 1]"}}}
'''

example3 = '''
Caption: "I adjusted my cellphone and continued playing the ukulele."
video_id: "0001_41641",
Action json:{"video_id": "0001_41641", "sequential descriptions": [ "person A adjust cellphone B", "person A play ukulele C" ], "time stamps": { "1": { "decription": [ "person A adjust cellphone B" ], "programmatic": [ "adjust(A, B)", "name(A, person)", "name(B, cellphone)" ], "duration": "short", "duration precise": "1/4", "video location": "early", "video location precise": "[0, 1/4]" }, "2": { "decription": [ "person A play ukulele C" ], "programmatic": [ "play(A, C)", "name(A, person)", "name(B, ukulele)" ], "duration": "long",  "duration precise": "3/4", "video location": "late", "video location precise": "[1/4, 1]"}}}
'''

example4 = '''
Caption: "A woman is teasing a kitten with a piece of meat, and the kitten is peeking its head from a chair to look at the meat."
Video_id: "0010_86102",
Action json: { "video_id": "0010_86102", "sequential descriptions": [ "woman A teasing kitten B with meat C", "kitten B peek at meat C from a chair D", ], "time stamps": { "1": { "decription": [ "woman A teasing kitten B with meat C", ], "programmatic": [ "sitting on(B, D)", "name(B, cat)", "name(D, chair)", "name(A, adult)", "name(C, meat)" ], "duration": "mid", "duration precise": "1/2", "video location": "early", "video location precise": "[0, 1/2]" }, "2": { "decription": [ "kitten B peek at meat C from a chair D" ], "programmatic": [ "catching(B, C)", "sitting on(B, D)" ], "duration": "long", "duration precise": "1", "video location": "mid", "video location precise": "[0, 1]"}}}
'''


example5 = '''
Caption: "The young boy receives another gift and sits on the floor."
Video_id: "03f2ed96-1719",
Action json: {"video_id": "03f2ed96-1719","sequential descriptions": ["boy A receives gift B", "boy A sits on the floor C"], "time stamps": {"1": { "decription": ["boy A receives gift B",], "programmatic": ["holding(B, D)", "name(A, boy)", "name(B, gift)",], "duration": "mid","duration precise": "1/2","video location": "early", "video location precise":  "[0, 1/2]"}, "2": { "decription": [ "boy A sits on the floor C",], "programmatic": ["sitting on(A, C)", "name(A, boy)", "name(C, floor)", ], "duration": "mid", "duration precise": "1/2", "video location": "late", "video location precise": "[1/2, 1]"}}}
'''

from pydantic import BaseModel, PositiveInt, TypeAdapter
from typing import List, Dict

class TimestampDescription(BaseModel):
    event_id: PositiveInt
    decription: list[str]
    programmatic: list[str]
    duration: str
    duration_precise: str
    video_location: str
    video_location_precise: list[str]
    
    class Config:
        schema_extra = {
            "required": ["event_id", "description", "programmatic", "duration", "duration_precise", "video_location", "video_location_precise"]
        }
    
class CaptionDescription(BaseModel):
    video_id: str
    sequential_descriptions: list[str]
    events: list[TimestampDescription]
    
    class Config:
        schema_extra = {
            "required": ["video_id", "sequential_descriptions", "events"]
        }
        
class BatchedCaptionDescription(BaseModel):
    data: Dict[str, CaptionDescription]
    
    class Config:
        schema_extra = {
            "required": ["data"]
        }
        
query = '''
Note all the relations are name, unary or binary.
A name relation takes in two arguments, the first is always a variable, and the second argument could be noun ("apple"), noun phrase ("ancient_building"), location("dark_forest"), etc. For example "name(A, "apple")" means the variable A refers to an apple. Please ensure no space occur in the second argument.
A unary relation takes in one variable as its argument. For example, close(A) means A is close to the camera.
A binary relation takes in two variables as its arguments. For example, above(A, B) means A is above B.
The entity in the binary and unary relation are variables in the form of captalized letters (A, B).
The predicate of the unary relation can be adjectives, verbs, and name.
The predicate of the binary relation can be preposition, and verb. 
Please include the name relation any time if applicable.
Please make the preposition and adjectives into seperate two relation. 
For each time stamp, please only describe the events that are happening at the same time, if any sequential events occur, put them into multiple time stamps. 
For example, instead of 'person A enter from left, person A walk to center, person A move to couch' in one time stamp, 
put it into three different time stamps: "person A enter from left", "person A walk to center", "person A move to couch".
Please only describe one single event in sequential description per time stamp. 
Please use as many relations as possible to precisely describe the action.
Please generate the action json programs for the following captions in the following format:
{"actions": {caption_id: action json programs}}
IMPORTANT: Please REMOVE all new line characters and extra spaces in the generated json!
'''

all_examples = [example1, example4, example5]
few_shot_prompt = '\n'.join(all_examples)
prompt = '\n'.join([user, context, few_shot_prompt, query])

def wrap_prompt(caption_ls, few_shot=True):
    output_prompt = [context]
    if few_shot:
        output_prompt.append(few_shot_prompt)

    output_prompt.append(query)

    for cid, (vid, caption) in enumerate(caption_ls):
        output_prompt.append(f"{cid}. video id: {vid} caption: {caption}")
        
    # for cid, caption in enumerate(caption_ls):
    #     output_prompt.append(f"{cid}. caption: {caption}")

    return '\n'.join(output_prompt)

if __name__ == "__main__":

    print(prompt)