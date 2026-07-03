import redis
import hashlib


class Cache:

    def __init__(self):
        self.r = redis.Redis(host="localhost", port=6379, decode_responses=True)

    def key(self, text):
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text):
        return self.r.get(self.key(text))

    def set(self, text, value):
        self.r.set(self.key(text), value, ex=3600)