"""Create a ChatVectorDBChain for question/answering."""
from langchain.callbacks.base import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains.llm import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

SystemMessagePrompt = "You are ChatGPT, a large language model trained by OpenAI to have friendly conversations with humans."
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SystemMessagePrompt),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])


memory = ConversationBufferMemory(return_messages=True)

def get_chain(
    question_handler, stream_handler, tracing: bool = False
) -> ConversationChain:
    """Create a ConversationChain for question/answering."""
    
    manager = AsyncCallbackManager([])
    stream_manager = AsyncCallbackManager([stream_handler])

    streaming_llm = ChatOpenAI(
        streaming=True, 
        callback_manager=stream_manager,
        verbose=True, 
        temperature=0,
        model='gpt-4', #'gpt-3.5-turbo'
        )
        
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(
        memory=memory, 
        prompt=prompt, 
        llm=streaming_llm,
        callback_manager=manager
        )
    return conversation
