from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import pandas as pd
NUM_SHOTS=5
qa_pairs = qa_pairs = pd.read_json("test_qa_pairs.jsonl", lines=True)
def compose_few_shot_prompt(qa_pairs):
    example_prompt = "Identify if {question}.\nText: {previous_text}\nClaim: {annotation}\nText continued: {next_text}\nAnswer: {answer}"
    single_shot_prompt = PromptTemplate(
        input_variables=["question", "previous_text", "annotation", "next_text", "answer"],
        template=example_prompt
    )
    example_qa_pairs = qa_pairs.loc[:NUM_SHOTS].to_dict(orient="records")
    few_shot_prompt = FewShotPromptTemplate(
        examples=example_qa_pairs,
        example_prompt=single_shot_prompt,
        prefix="Here are some examples of identifying whether a hypothesis is entailed or contradicted by a given text:\n",
        suffix="Given the following text and hypothesis, identify if the hypothesis is entailed or contradicted, only output Entailment or Contradicted, no other words.\n Hypothesis: {question}\nText: {previous_text}\nClaim: {annotation}\nText continued: {next_text}\nAnswer:",
        input_variables=["question", "previous_text", "annotation", "next_text"],
        example_separator="\n\n"
    )

    # RAG PROMPT
    system_prompt = "Here are some resources that are maybe helpful to you. {context}\n Given the following text and hypothesis, identify if the hypothesis is entailed or contradicted, only output Entailment or Contradicted, no other words."
    human_prompt = """
    Question starts here:
    {input}
    """
    chat_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])
    return chat_template




#