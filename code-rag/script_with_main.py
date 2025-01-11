import os
from typing import List

from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import TypedDict

MAX_ITERATIONS = 3
REFLECT = False


def load_docs():
    """Load and process LCEL documentation."""
    url = "https://python.langchain.com/docs/concepts/lcel/"
    loader = RecursiveUrlLoader(url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text)
    docs = loader.load()
    d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
    d_reversed = list(reversed(d_sorted))
    return "\n\n\n --- \n\n\n".join([doc.page_content for doc in d_reversed])


class Code(BaseModel):
    """Schema for code solutions to questions about LCEL."""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
    """

    error: str
    messages: List
    generation: str
    iterations: int


def generate(state: GraphState):
    """Generate a code solution."""
    print("---GENERATING CODE SOLUTION---")
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]

    if error == "yes":
        messages += [
            (
                "user",
                "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:",
            )
        ]

    code_solution = code_gen_chain.invoke({"context": concatenated_content, "messages": messages})
    messages += [
        (
            "assistant",
            f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}",
        )
    ]

    iterations = iterations + 1
    return {"generation": code_solution, "messages": messages, "iterations": iterations}


def code_check(state: GraphState):
    """Check code execution."""
    print("---CHECKING CODE---")
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    imports = code_solution.imports
    code = code_solution.code

    try:
        exec(imports)
    except Exception as e:
        print("---CODE IMPORT CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the import test: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    try:
        exec(imports + "\n" + code)
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the code execution test: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    print("---NO CODE TEST FAILURES---")
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }


def reflect(state: GraphState):
    """Reflect on errors."""
    print("---GENERATING CODE SOLUTION---")
    messages = state["messages"]
    iterations = state["iterations"]
    code_solution = state["generation"]

    reflections = code_gen_chain.invoke({"context": concatenated_content, "messages": messages})
    messages += [("assistant", f"Here are reflections on the error: {reflections}")]
    return {"generation": code_solution, "messages": messages, "iterations": iterations}


def decide_to_finish(state: GraphState):
    """Determines whether to finish."""
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == MAX_ITERATIONS:
        print("---DECISION: FINISH---")
        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        return "reflect" if REFLECT else "generate"


def setup_workflow():
    """Set up the workflow graph."""
    workflow = StateGraph(GraphState)

    workflow.add_node("generate", generate)
    workflow.add_node("check_code", code_check)
    workflow.add_node("reflect", reflect)

    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "check_code")
    workflow.add_conditional_edges(
        "check_code",
        decide_to_finish,
        {
            "end": END,
            "reflect": "reflect",
            "generate": "generate",
        },
    )
    workflow.add_edge("reflect", "generate")
    return workflow.compile()


def main():
    global code_gen_chain, concatenated_content

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # Load documentation
    concatenated_content = load_docs()

    # Setup code generation chain
    code_gen_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a coding assistant with expertise in LCEL, LangChain expression language. \n 
    Here is a full set of LCEL documentation:  \n ------- \n  {context} \n ------- \n Answer the user 
    question based on the above provided documentation. Ensure any code you provide can be executed \n 
    with all required imports and variables defined. Structure your answer with a description of the code solution. \n
    Then list the imports. And finally list the functioning code block. Here is the user question:""",
            ),
            ("placeholder", "{messages}"),
        ]
    )

    llm = ChatOpenAI(temperature=0, model="gpt-4")
    code_gen_chain = code_gen_prompt | llm.with_structured_output(Code)

    # Setup and run workflow
    app = setup_workflow()

    # Example usage
    # question = "How can I directly pass a string to a runnable and use it to construct the input needed for my prompt?"
    question = "How can I use a prompt and model to create a chain in LCEL that returns raw ChatMessages?"
    result = app.invoke({"messages": [("user", question)], "iterations": 0, "error": ""})

    return result["generation"]


if __name__ == "__main__":
    result = main()
    print("\nFinal Solution:")
    print(result.prefix)
    print()
    print(result.imports)
    print()
    print(result.code)
