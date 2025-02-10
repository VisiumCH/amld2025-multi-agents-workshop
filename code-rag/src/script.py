# %%

import os
import pickle
from dataclasses import dataclass, field
from typing import Annotated, Literal, cast

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field

# Ensure API key is set.
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY environment variable")

MAX_ITERATIONS = 3

# %%
# --- Document loading & retrieval setup ---
docs = pickle.load(open("docs.pkl", "rb"))


def split_docs(docs_list: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=100, disallowed_special=()
    )
    return splitter.split_documents(docs_list)


def create_vstore(docs_splits: list[Document]) -> Chroma:
    Chroma().reset_collection()

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma.from_documents(documents=docs_splits, embedding=embedding_model)


def reduce_docs(docs_list: list[Document]) -> list[Document]:
    seen = set()
    unique = []
    for doc in docs_list:
        if doc.id not in seen:
            seen.add(doc.page_content)
            unique.append(doc)
    return unique


# %%
docs_splits = split_docs(docs[:10])

# %%
vstore = create_vstore(docs_splits)

# %%
retriever = vstore.as_retriever(search_kwargs={"k": 2})


# --- Schema for structured code generation ---


class Code(BaseModel):
    prefix: str = Field(description="A description of the problem and proposed approach.")
    code: str = Field(description="The python code block. It should be runnable and solve the problem.")


class DecisionSchema(BaseModel):
    prefix: str = Field(description="A description of the the decision process.")
    route: Literal["generate", "retrieve"] = Field(description="The decision to generate or retrieve.")
    retrieval_query: str | None = Field(description="The query for retrieving additional documentation.")


# --- State Definitions ---


@dataclass(kw_only=True)
class AgentState:
    messages: Annotated[list[str], add_messages] = field(default_factory=list)
    retrieval_query: str | None = field(default=None)
    documents: list[Document] = field(default_factory=list)
    iterations: int = field(default=0)


# --- Workflow Nodes using Command ---


# --- Retrieve Node ---
def retrieve_node(state: AgentState) -> Command[Literal["generate"]]:
    print(f"Retrieving documents for query: {state.retrieval_query}")
    docs_found = retriever.invoke(state.retrieval_query)

    return Command(
        goto="generate",
        update={
            "documents": docs_found,
        },
    )


# --- Decision Node ---


DECISION_PROMPT = """

You are a smart decision maker. You have access to a vector database containing the latest LangGraph documentation.

Based on the conversation, any error messages, and the current code solution (if any), decide whether documentation should be retrieved or if the system should generate a solution right away.

If you think additional documentation is needed, respond with 'retrieve' and provide a retrieval query to search for more information.

If you think a solution can be generated based on the current context, respond with 'generate'.

Previously, you made the following query:

<retrieval_query>
{retrieval_query}
</retrieval_query>

This returned the following documents:

<documentation>
{documentation}
</documentation>

"""

decision_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(DECISION_PROMPT),
        MessagesPlaceholder("messages"),
    ]
)
decision_model = ChatOpenAI(temperature=0, model="gpt-4o").with_structured_output(DecisionSchema)
decision_chain = decision_prompt | decision_model


def decision_node(state: AgentState) -> Command[Literal["generate", "retrieve", END]]:
    if state.iterations >= MAX_ITERATIONS:
        return Command(goto=END)

    print(f"Decision node with {len(state.documents)} documents.")
    documentation = "\n".join(doc.page_content for doc in state.documents)

    print(state.retrieval_query)

    decision_value = decision_chain.invoke(
        {
            "retrieval_query": state.retrieval_query,
            "documentation": documentation,
            "messages": state.messages,
        }
    )

    decision_value = cast(DecisionSchema, decision_value)  # TODO: repetition but avoids linting warnings

    print(f"Decision: {decision_value.route}")
    print(f"Prefix: {decision_value.prefix}")

    if decision_value.route == "retrieve":
        print(f"Retrieval query: {decision_value.retrieval_query}")
        return Command(
            goto="retrieve",
            update={"retrieval_query": decision_value.retrieval_query},
        )
    else:  # decision_value.route == "generate":
        return Command(
            goto="generate",
        )


# --- Code Generation Node ---


GENERATION_PROMPT = """

You are a coding assistant with expertise in LangChain and LangGraph. You are tasked with generating a code solution based on the provided documentation.

You should ouput a runnable code solution with imports that solves the problem described in the documentation.

<documentation>
{documentation}
</documentation>

"""

code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(GENERATION_PROMPT),
        MessagesPlaceholder("messages"),
    ]
)
code_gen_model = ChatOpenAI(temperature=0, model="gpt-4o").with_structured_output(Code)
code_gen_chain = code_gen_prompt | code_gen_model


def generate_node(state: AgentState) -> Command[Literal["test"]]:
    print(f"Generating code solution with {len(state.documents)} documents.")
    documentation = "\n".join(doc.page_content for doc in state.documents)

    code_solution = code_gen_chain.invoke(
        {
            "documentation": documentation,
            "messages": state.messages,
        },
    )

    code_solution = cast(Code, code_solution)  # TODO: repetition but avoids linting warnings

    return Command(
        goto="test",
        update={
            "messages": AIMessage(code_solution.code),
            "iterations": state.iterations + 1,
        },
    )


def test_node(state: AgentState) -> Command[Literal["decision", "critique", END]]:
    print(f"Testing code solution with {len(state.documents)} documents.")

    code = state.messages[-1]

    try:
        exec(code)
    except Exception as e:
        return Command(
            goto="decision",
            update={
                "messages": HumanMessage(f"Code execution failed: {e}"),
            },
        )

    return Command(
        goto="critique",
        update={
            "messages": "Code execution successful.",
        },
    )


@dataclass(kw_only=True)
class CritiqueSchema(BaseModel):
    decision: Literal["decision", "end"] = Field(description="The critique decision.")
    critique: str = Field(description="The critique feedback.")


CRITIQUE_PROMPT = """
    You are a critique assistant. Based on the provided messages and context, critique the generated code solution.

    Give suggestions to the code generator for improvement and provide feedback on the code quality.

    You must decide whether the code solution is acceptable or if it needs improvement to conform to the user request.

    Don't be to strict, we are aiming for a quick solution, but make sure it is correct.

    If you think the solution is acceptable, respond with 'end'.

    If you think the solution must be improved to better fit the user request, respond with 'decision' and provide feedback on the code.

"""

critique_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(CRITIQUE_PROMPT),
        MessagesPlaceholder("messages"),
    ]
)
critique_model = ChatOpenAI(temperature=0, model="gpt-4o").with_structured_output(CritiqueSchema)
critique_chain = critique_prompt | critique_model


def critique_node(state: AgentState) -> Command[Literal["decision", END]]:
    critique_response = critique_chain.invoke({"messages": state.messages})

    critique_response = cast(CritiqueSchema, critique_response)

    if critique_response.decision == "end":
        return Command(goto=END)
    else:
        return Command(
            goto="decision",
            update={
                "messages": HumanMessage(f"{critique_response.critique}"),
            },
        )


# --- Workflow Setup ---
def setup_workflow():
    builder = StateGraph(AgentState)
    builder.add_edge(START, "decision")
    builder.add_node("decision", decision_node)
    builder.add_node("generate", generate_node)
    builder.add_node("test", test_node)
    builder.add_node("critique", critique_node)
    builder.add_node("retrieve", retrieve_node)

    # Other edges are added dynamically based with Command objects!

    return builder.compile()


app = setup_workflow()

# --- Terminal run ---
if __name__ == "__main__":
    # --- Compile the Workflow ---
    app.invoke({"messages": HumanMessage("Implement a simple RAG system in Langgraph")})
