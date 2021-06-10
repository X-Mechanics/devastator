import json

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def parameters(self):
        return self.__dict__

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.__dict__, indent=2) + "\n"

    def save_json(self, path: str):
        with open(path, 'w') as f:
            f.write(self.to_json_string())

    @classmethod
    def from_json(cls, path: str):
        with open(path, 'r') as f:
            cfg = cls(**json.loads(f.read()))
            return cfg