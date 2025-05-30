import redis
import pickle
from collections import deque 

class order(): 
    def __init__(self, number):  
        self.order_number = number 
    def __repr__(self):
        return f'order({self.order_number})'
    def __eq__(self, other):
        return isinstance(other, order) and self.order_number == other.order_number 

class SharedOrderQ:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_key='shared_orderQ'): 

        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        self.redis_key = redis_key 

        self.redis.delete(self.redis_key)

        self.coffee = deque()
        self.labels = deque()
        self.writing = None
        self.brewing = None

        # Try loading existing state
        self.load_from_redis()

    def add_order(self, order):
        self.coffee.append(order)
        self.labels.append(order)
        self.save_to_redis()

    def next_label(self):
        if len(self.labels) > 0:
            self.writing = self.labels.popleft()
            self.save_to_redis()
            return self.writing
        return None

    def next_coffee(self):
        if len(self.coffee) > 0:
            self.brewing = self.coffee.popleft()
            self.save_to_redis()
            return self.brewing
        return None

    def save_to_redis(self):
        data = pickle.dumps({
            'coffee': list(self.coffee),
            'labels': list(self.labels),
            'writing': self.writing,
            'brewing': self.brewing
        })
        self.redis.set(self.redis_key, data)

    def load_from_redis(self):
        data = self.redis.get(self.redis_key)
        if data:
            obj = pickle.loads(data)
            self.coffee = deque(obj['coffee'])
            self.labels = deque(obj['labels'])
            self.writing = obj['writing']
            self.brewing = obj['brewing']
