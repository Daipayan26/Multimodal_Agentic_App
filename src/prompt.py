from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import (
                                    SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate,
                                    PromptTemplate,
                                    ChatPromptTemplate,
                                    MessagesPlaceholder
                                    )

from langchain_core.runnables.history import RunnableWithMessageHistory

system_prompt = (
    "You are an Food Nutrition assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.Dont't use based on provided context in the response only return related answers."
    "\n\n"
    "{context}"
)
human=HumanMessagePromptTemplate.from_template("{input}")
messages=[system_prompt,MessagesPlaceholder(variable_name='history'),human]

prompt=ChatPromptTemplate(messages=messages)
