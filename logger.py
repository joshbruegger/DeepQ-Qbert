class Logger:
    def __init__(self, dict: dict = None):
        self.dict = dict if dict else {}

    def log(self, key, time, value):
        if key not in self.dict:
            self.dict[key] = []
        self.dict[key].append((time, value))

    def get_data(self, key):
        return self.dict[key]

    def clear(self):
        self.dict = {}

    def save_plot(self, dir: str, name: str):
        pass
