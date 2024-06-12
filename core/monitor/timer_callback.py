from langchain_core.callbacks import BaseCallbackHandler
import time


class TimerCallback(BaseCallbackHandler):
    def __init__(self, time_llm, time_record):
        self.time_record = time_record
        self.time_llm = time_llm
        self.input_token=0
        self.output_token=0

    def on_chain_start(self, s1, s2, **kwargs):
        if len(kwargs["tags"]) > 0:
            self.time_record[kwargs["tags"][0]] = time.perf_counter()
        if kwargs['name'] == 'PromptTemplate':
            self.input_token = s2['context'].split().__len__() + s2['input'].split().__len__()
        elif kwargs['name'] == 'StrOutputParser':
            self.output_token = s2.split().__len__()

    def on_chain_end(self, s1, **kwargs):
        if len(kwargs["tags"]) > 0:
            self.time_record[kwargs["tags"][0]] = (
                time.perf_counter() - self.time_record[kwargs["tags"][0]]
            )

    def on_llm_start(self, s1, s2, **kwargs):
        self.time_llm[0] = time.perf_counter()

    def on_llm_end(self, re, **kwargs):
        self.time_llm[0] = time.perf_counter() - self.time_llm[0]
