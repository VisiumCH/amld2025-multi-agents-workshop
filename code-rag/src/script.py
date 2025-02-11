# %%

import os
import pickle
from dataclasses import dataclass, field
from typing import Annotated, Literal, cast

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Environment and Global Configuration
# -----------------------------------------------------------------------------

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# Workflow Configuration
MAX_ITERATIONS = 10
RETRIEVAL_K = 2
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

# -----------------------------------------------------------------------------
# Utility Functions for Document Handling
# -----------------------------------------------------------------------------


def load_documents(pickle_filepath: str = "docs.pkl") -> list[Document]:
    """Load documents from a pickle file."""
    with open(pickle_filepath, "rb") as file:
        return pickle.load(file)


def split_documents(documents: list[Document]) -> list[Document]:
    """Split each document into chunks using a recursive text splitter."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=100, disallowed_special=()
    )
    return splitter.split_documents(documents)


def initialize_vector_store(document_chunks: list[Document]) -> Chroma:
    """Reset the Chroma collection and initialize a vector store using document chunks."""
    Chroma().reset_collection()
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma.from_documents(documents=document_chunks, embedding=embedding_model)


def deduplicate_documents(documents: list[Document]) -> list[Document]:
    """Remove duplicate documents based on their unique identifier."""
    unique_docs = {doc.id: doc for doc in documents}
    return list(unique_docs.values())


# -----------------------------------------------------------------------------
# Document Preparation and Retriever Setup
# -----------------------------------------------------------------------------

# For demonstration, load and process a subset of documents.
loaded_documents = load_documents("docs.pkl")
chunked_documents = split_documents(loaded_documents)

# Initialize vector store and create a retriever.
vector_store = initialize_vector_store(chunked_documents)
retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

# -----------------------------------------------------------------------------
# Structured Output Schemas for the Workflow
# -----------------------------------------------------------------------------


class RetrievalDecision(BaseModel):
    """Retrieval decision to either fetch more docs or generate a solution."""

    rationale: str = Field(description="Explanation for the decision")
    decision: Literal["generation", "retrieval"] = Field(
        description="Indicates whether to generate a solution or retrieve docs"
    )
    retrieval_query: str | None = Field(description="Query to retrieve additional documentation, if needed")


class GeneratedSolution(BaseModel):
    """Generated solution with rationale and code."""

    rationale: str = Field(description="Explanation of the solution approach")
    code: str = Field(description="The complete Python code solution")


class ReviewDecision(BaseModel):
    """Review outcome indicating if the solution is accepted or needs revision."""

    rationale: str = Field(description="Feedback explaining the review decision")
    decision: Literal["revise", "accept"] = Field(description="Indicates whether to revise or accept the solution")


# -----------------------------------------------------------------------------
# RAG Workflow State Definition
# -----------------------------------------------------------------------------


@dataclass(kw_only=True)
class GraphState:
    """Workflow state: messages, query, docs, solution, and iteration count."""

    messages: Annotated[list[BaseMessage], add_messages] = field(default_factory=list)
    retrieval_query: str | None = field(default=None)
    retrieved_documents: list[Document] = field(default_factory=list)
    generated_solution: str | None = field(default=None)
    iterations: int = field(default=0)


# -----------------------------------------------------------------------------
# System Prompts
# -----------------------------------------------------------------------------

RETRIEVAL_DECISION_PROMPT = """
You are a Retrieval Decision Agent in a LangGraph Retrieval-Augmented Generation system with access to a large corpus of LangGraph documentation in a vector store.

Your role is to determine if the information at hand is sufficient to generate a code solution, or if further documentation must be retrieved.

The latest retrieval query is:
{retrieval_query}

and the retrieved documentation is:
{documentation}

Carefully assess whether whether the available context is sufficient for generating a LangGraph solution to the user request.

If you perceive that crucial details are missing, you should return 'retrieval' along with a retrieval query to prompt further data retrieval. Otherwise, return 'generation' to proceed with the code generation phase.

Include a brief rationale explaining your decision.
"""

SOLUTION_GENERATION_PROMPT = """
You are a Generation Agent in a LangGraph code generation workflow. Your task is to produce a Python solution that implements the requested LangGraph workflow.

The current documentation available for generating the solution is:
{documentation}

Based on your expertise, the documentation, and the current dialogue context, craft a Python code solution that effectively addresses the user request.

Your solution should be clear, concise, and well-structured, ensuring that it is both functional and adheres to best practices.

Return a rationale explaining your solution approach along with the complete Python code solution.
"""

SOLUTION_REVIEW_PROMPT = """
You are a Review Agent in a LangGraph code generation workflow. Your role is to evaluate the generated Python solution and determine if it meets the user's request and the LangGraph standards.

Review the dialogue context to assess that the proposed solution is technically sound, adheres to LangGraph best practices and fulfills the user's requirements.

If the solution meets the standards of a reliable LangGraph implementation, confirm its acceptance with 'accept'.

Otherwise, prompt further refinement by returning 'revise' along with your rationale for the necessary changes.

Your rationale should provide constructive and well justified feedback to guide the revision process.
"""


# -----------------------------------------------------------------------------
# Workflow Node Implementations
# -----------------------------------------------------------------------------

# ----- Retrieval Node -----


def retrieval_node(state: GraphState) -> Command[Literal["generation"]]:
    """Retrieve documentation based on the current retrieval query."""
    assert state.retrieval_query, "No retrieval query provided."
    retrieved_docs = retriever.invoke(state.retrieval_query)
    return Command(
        goto="generation",
        update={"retrieved_documents": retrieved_docs},
    )


# ----- Retrieval Decision Node -----

retrieval_decision_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(RETRIEVAL_DECISION_PROMPT),
        MessagesPlaceholder("messages"),
    ]
)
retrieval_decision_model = ChatOpenAI(temperature=LLM_TEMPERATURE, model=LLM_MODEL).with_structured_output(
    RetrievalDecision
)
retrieval_decision_chain = retrieval_decision_prompt_template | retrieval_decision_model


def retrieval_decision_node(state: GraphState) -> Command[Literal["generation", "retrieval", END]]:
    """Decide to generate a solution or retrieve more documentation, with iteration limit enforcement."""
    # Enforce the maximum iterations limit.
    if state.iterations >= MAX_ITERATIONS:
        return Command(
            goto=END,
            update={
                "messages": AIMessage("Maximum iteration limit reached. Finalizing workflow."),
            },
        )

    documentation = "\n".join(doc.page_content for doc in state.retrieved_documents) or None

    retrieval_decision = retrieval_decision_chain.invoke(
        {
            "retrieval_query": state.retrieval_query,
            "documentation": documentation,
            "messages": state.messages,
        }
    )
    retrieval_decision = cast(RetrievalDecision, retrieval_decision)

    if retrieval_decision.decision == "retrieval":
        return Command(
            goto="retrieval",
            update={
                "retrieval_query": retrieval_decision.retrieval_query,
            },
        )
    else:
        return Command(goto="generation")


# ----- Generation Node -----

generation_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SOLUTION_GENERATION_PROMPT),
        MessagesPlaceholder("messages"),
    ]
)
generation_model = ChatOpenAI(temperature=LLM_TEMPERATURE, model=LLM_MODEL).with_structured_output(GeneratedSolution)
generation_chain = generation_prompt_template | generation_model


def generation_node(state: GraphState) -> Command[Literal["testing"]]:
    """Generate a Python code solution based on the current enriched context."""
    documentation_summary = "\n".join(doc.page_content for doc in state.retrieved_documents)
    solution_output = generation_chain.invoke(
        {
            "documentation": documentation_summary,
            "messages": state.messages,
        }
    )
    solution_output = cast(GeneratedSolution, solution_output)
    return Command(
        goto="testing",
        update={
            "generated_solution": solution_output.code,
            "messages": AIMessage(solution_output.code),
            "iterations": state.iterations + 1,
        },
    )


# ----- Testing Node -----


def testing_node(state: GraphState) -> Command[Literal["retrieval_decision", "review", END]]:
    """Test the generated solution by executing the code."""
    code = state.generated_solution
    assert code, "No code solution provided."

    try:
        # exec_globals: dict = {}
        exec(code)  # , exec_globals)
    except Exception as error:
        error_message = f"Code execution failed: {error}"
        return Command(
            goto="retrieval_decision",
            update={"messages": HumanMessage(error_message)},
        )

    success_message = "Code execution successful."
    return Command(
        goto="review",
        update={"messages": HumanMessage(success_message)},
    )


# ----- Review Node -----

review_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SOLUTION_REVIEW_PROMPT),
        MessagesPlaceholder("messages"),
    ]
)
review_model = ChatOpenAI(temperature=LLM_TEMPERATURE, model=LLM_MODEL).with_structured_output(ReviewDecision)
review_chain = review_prompt_template | review_model


def review_node(state: GraphState) -> Command[Literal["retrieval_decision", END]]:
    """Review the generated solution and determine if it is acceptable."""
    review_result = review_chain.invoke({"messages": state.messages})
    review_result = cast(ReviewDecision, review_result)
    if review_result.decision == "accept":
        return Command(goto=END)
    else:
        return Command(
            goto="retrieval_decision",
            update={"messages": HumanMessage(review_result.rationale)},
        )


# -----------------------------------------------------------------------------
# Workflow Setup and Main Execution
# -----------------------------------------------------------------------------


def setup_workflow() -> CompiledStateGraph:
    """Set up and compile the state graph workflow."""
    graph_builder = StateGraph(GraphState)
    graph_builder.add_edge(START, "retrieval_decision")
    graph_builder.add_node("retrieval_decision", retrieval_decision_node)
    graph_builder.add_node("generation", generation_node)
    graph_builder.add_node("testing", testing_node)
    graph_builder.add_node("review", review_node)
    graph_builder.add_node("retrieval", retrieval_node)
    # Additional edges are added dynamically via Command objects.
    return graph_builder.compile()


app = setup_workflow()