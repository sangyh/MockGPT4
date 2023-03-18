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

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = ChatOpenAI(temperature=0, model='gpt-4')

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
