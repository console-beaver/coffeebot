from collections import deque

class orderQ(): 
    def __init__(self): 
        self.coffee = deque() 
        self.labels = deque()   

        self.writing = None 
        self.brewing = None
    
    def add_order(self, order): 
        self.coffee.append(order) 
        self.labels.append(order)  

    def next_label(self): 
        if len(self.labels) > 0: 
            self.writing = self.labels.popleft() 
            return self.writing 
        else: 
            return None  
        
    def next_coffee(self): 
        if len(self.coffee) > 0: 
            self.brewing = self.coffee.popleft() 
            return self.brewing 
        else: 
            return None 

class order(): 
    def __init__(self, number):  
        self.order_number = number

class waiter_state():  
    """
    This class is used to manage and determine the state of the waiter. 
    ========================STATES========================================== 
    - init: The initial state of the waiter 
        - on start up 
    - collecting_order: The waiter is collecting an order one at a time for some time 
        - the writing label queue is empty
    - writing_label: The waiter is writing a label for the order  
        - the writing label queue is not empty
    """ 
    def __init__(self): 
        self.state = 'init'  
    
    def compute_state(self, queue): 
        if len(queue.labels) == 0:   
            self.state = 'collecting_order' 
        elif len(queue.labels) > 0: 
            self.state = 'writing_label' 

class barista_state(): 
    """
    This class is used to manage and determine the state of the barista. 
    ========================STATES========================================== 
    - init: The initial state of the barista 
        - on start up 
    - brewing: The barista is brewing coffee for some time 
        - the brewing coffee queue is not empty
    - done_brewing: The barista is done brewing coffee for some time 
        - the brewing coffee queue is empty
    """ 
    def __init__(self): 
        self.state = 'init'  
    
    def compute_state(self, queue): 
        if len(queue.coffee) > 0:   
            self.state = 'brewing' 
        elif len(queue.coffee) == 0: 
            self.state = 'done_brewing'
