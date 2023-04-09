from chat_with_x.processing.query_data import get_chain
from chat_with_x.processing.ingest_data import create_vectorstore_from_local_file
from misc.utils import retrieve_local_api_keys, load_pickle_file, print_colored_output
import argparse
import os


def chat_completion_script(f_path, file_desc, chunk_size=1024, chunk_overlap=128, output_fpath=None, no_save=False,
                           model_name="gpt-3.5-turbo", task="initial", instruction="conversational", temperature=0.0,
                           use_color=False, input_clr="blue", response_clr="green", system_clr="magenta",
                           search_type="similarity", **kwargs):
    """ TBD """

    # Load the API keys from the .env file
    retrieve_local_api_keys()

    # Create the vectorstore from the input file (or load pickle if passed)
    if f_path.endswith(".pkl") or f_path.endswith(".pickle"):
        vs = load_pickle_file(f_path)
    else:
        vs, _ = create_vectorstore_from_local_file(f_path,
                                                   chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   no_save=no_save,
                                                   output_fpath=output_fpath)

    # Get the chain
    chain = get_chain(
        vs, file_desc,
        task_type=task,
        instruction=instruction,
        model_name=model_name,
        search_type=search_type,
        model_temperature=temperature,
    )

    # Chat loop
    chat_history = []
    print("\n... CHAT-WITH-X ...\n")
    while True:
        input_text = input("\nEnter your message (or press enter to exit):\n")
        if not input_text: break

        result = chain({"question": input_text, "chat_history": chat_history})
        chat_history.append((input_text, result["answer"]))

        # Print the colored output
        print_colored_output(
            input_text, result["answer"], system_message=", ".join([task, instruction]),
            input_color=input_clr, response_color=response_clr, system_color=system_clr, full_color=use_color
        )


def main():
    parser = argparse.ArgumentParser(description="Generate a conversation with memory using LangChain ChatOpenAI agent")
    parser.add_argument("f_path", help="Path to the file to be loaded")
    parser.add_argument("--file_desc", help="One line file description")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Chunk size to be used when splitting the text")
    parser.add_argument("--chunk_overlap", type=int, default=128, help="The amount of overlap between chunks")
    parser.add_argument("--output_fpath", help="The directory to save the vectorstore to. If not specified, "
                                               "the vectorstore will be saved at the location of the input file as a "
                                               "pickle file")
    parser.add_argument("--no_save", action="store_true", help="Whether to save the vectorstore to disk.")
    parser.add_argument("-n", "--model_name", type=str, default="gpt-3.5-turbo", help="Model name for the ChatLLM")

    parser.add_argument("-t", "--task", type=str, default="initial",
                        help="The type of task to be performed by the LLM")
    parser.add_argument("-i", "--instruction", type=str, default="conversational",
                        help="The instruction to be given to the LLM")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for randomness in output")
    parser.add_argument("-c", "--use_color", action="store_false", help="Color only to titles")
    parser.add_argument("--input_clr", type=str, default="blue", help="Color for user input text")
    parser.add_argument("--response_clr", type=str, default="green", help="Color for AI response text")
    parser.add_argument("--system_clr", type=str, default="magenta", help="Color for system message text")
    parser.add_argument("--search_type", type=str, default="similarity",
                        help="The type of search to be performed. Either 'similarity' or 'mmr'")
    args = parser.parse_args()
    chat_completion_script(**args.__dict__)


if __name__ == "__main__":
    main()
