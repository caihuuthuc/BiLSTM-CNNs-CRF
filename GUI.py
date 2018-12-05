from tkinter import *

import grpc

# import the generated classes
import calculator_pb2
import calculator_pb2_grpc
import re
from tkinter.scrolledtext import ScrolledText

OUTSIDE_COLOR = 'white'
ADVERSEREACTOIN_COLOR = 'deep sky blue'
SEVERITY_COLOR = 'green yellow'
FACTOR_COLOR = 'gold'
DRUGCLASS_COLOR = 'hot pink'
ANIMAL_COLOR = 'medium orchid'
NEGATION_COLOR = 'tan1'

LIMIT_WIDTH = 80

FONT_SIZE = 20

def NER_analysis():
    global root
    global e1
    global frames
    current_row = 4
    sentence = e1.get("1.0",END).strip()
    # print(sentence)
    req = calculator_pb2.Req(sentence=sentence)
    response = stub.getNER(req)
    # print(response.label)
    # colors = [OUTSIDE_COLOR, ADVERSEREACTOIN_COLOR, SEVERITY_COLOR, FACTOR_COLOR, DRUGCLASS_COLOR, ANIMAL_COLOR, NEGATION_COLOR]
    colors = {'Outside': 'white', 
        'PER': 'deep sky blue', 
        'MISC': 'green yellow', 
        'ORG': 'gold', 
        'LOC': 'hot pink'}
    current_width = 0

    for frame in frames:
        frame.destroy()
    frames = []
    
    frames.append(Frame(root, width=LIMIT_WIDTH + 5))
    frames[0].grid(row=current_row)
    for word, tag in zip(re.split(' ', response.token), re.split(' ', response.label)):

        if tag == 'O':
            entity = 'Outside'
        else:
            entity = re.split('-', tag)[1].strip()
        if current_width + len(word) < LIMIT_WIDTH:
            Label(frames[-1], text=word, bg=colors[entity], width=len(word), font=("Helvetica", FONT_SIZE)).pack(side=LEFT)
            current_width += len(word)
        else:
            current_row += 1
            current_width = 0
            frames.append(Frame(root, width=LIMIT_WIDTH + 5))
            frames[-1].grid(row=current_row)
            Label(frames[-1], text=word, bg=colors[entity], width=len(word) + 2, font=("Helvetica", FONT_SIZE)).pack(side=LEFT)

    for idx, w in enumerate(['PER', 'MISC', 'LOC', 'ORG']):
        frames.append(Frame(root))
        frames[-1].grid(row=current_row + idx + 2, column = 0)
        Label(frames[-1], text=w, bg=colors[w], font=("Helvetica", FONT_SIZE)).grid(row=0)
    
def POS_analysis():
    global root
    global e1
    global frames
    current_row = 4
    sentence = e1.get("1.0",END).strip()
    # print(sentence)
    req = calculator_pb2.Req(sentence=sentence)
    response = stub.getPOS(req)


    current_width = 0
    for frame in frames:
        frame.destroy()
    frames = []
    
    frames.append(Frame(root, width=LIMIT_WIDTH + 5))
    frames[0].grid(row=current_row)
    for word, tag in zip(re.split(' ', response.token), re.split(' ', response.label)):
        entity = tag
        if current_width + max(len(word) + 1, len(entity) + 1) < LIMIT_WIDTH:
            Label(frames[-1], text="%s\n%s" % (word, entity), bg='white', width=max(len(word) + 1, len(entity) + 1), font=("Helvetica", FONT_SIZE)).pack(side=LEFT)
            # Label(frames[-1], text=entity, bg='white').pack(side=BOTTOM)
            current_width += max(len(word) + 1, len(entity) + 1)
        else:
            current_row += 1
            current_width = 0
            frames.append(Frame(root, width=LIMIT_WIDTH + 5))
            frames[-1].grid(row=current_row)
            Label(frames[-1], text="%s\n%s" % (word, entity), bg='white', width=len(word) + 2, font=("Helvetica", FONT_SIZE)).pack(side=LEFT)


if __name__ == '__main__':
    root = Tk()
    root.title('DEMO')
    root.geometry("1600x800+100+100")
    
    root_frame = Frame(root, height=25, width=110)
    Label(root_frame, text='Enter text', height=3, pady=5, font=("Helvetica", FONT_SIZE)).grid(row=0)
    root_frame.grid(row=0)
    frames = []

    e1 = Text(root_frame, width=80, height=10, pady=5, font=("Helvetica", FONT_SIZE))

    e1.grid(row=0, column=1)

    frame = Frame(root, width=110)
    frame.grid(row=1)

    Button(frame, text="NER Analysis", command=NER_analysis, pady=1, font=("Helvetica", FONT_SIZE)).grid(row=1, column=1)
    Button(frame, text="POS Analysis", command=POS_analysis, pady=1, font=("Helvetica", FONT_SIZE)).grid(row=1, column=2)


    # open a gRPC channel
    channel = grpc.insecure_channel('0.0.0.0:50051')

    # create a stub (client)
    stub = calculator_pb2_grpc.CalculatorStub(channel)
    
    

    root.mainloop()