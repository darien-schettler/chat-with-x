from dotenv import load_dotenv
from colorama import init, Fore, Style
from typing import Any
import pickle
import os


def load_pickle_file(file_path: str) -> Any:
    """
    Load a pickle file from the specified file path and return its contents.

    Parameters:
    file_path (str): The path to the pickle file to load.

    Returns:
    Any: The contents of the loaded pickle file.

    Raises:
    TypeError: If file_path is not a string.
    FileNotFoundError: If the file at file_path does not exist.
    """
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")

    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"file not found at path: {file_path}")


def save_pickle_file(file_path: str, data: Any) -> None:
    """
    Save the specified data as a pickle file at the specified file path.

    Parameters:
    file_path (str): The path to save the pickle file to.
    data (Any): The data to save in the pickle file.

    Raises:
    TypeError: If file_path is not a string.
    """
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")

    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def print_colored_output(input_text, response_text,
                         system_message=None,
                         input_color="blue",
                         response_color="green",
                         system_color="magenta",
                         full_color=False):
    """ Print the input and response text with the specified colors

    Args:
        input_text (str): The input text
        response_text (str): The response text
        system_message (str, optional): The system message text
        input_color (str, optional): The color to use for the input text
        response_color (str, optional): The color to use for the response text
        system_color (str, optional): The color to use for the system message text
        full_color (bool, optional): If True, color the entire string, otherwise color only the first part

    Returns:
        None; prints the input and response text with the specified colors
    """

    icc = getattr(Fore, input_color.upper(), Fore.RESET)
    rcc = getattr(Fore, response_color.upper(), Fore.RESET)
    scc = getattr(Fore, system_color.upper(), Fore.RESET)

    if full_color:
        if system_message:
            print(f"\n{scc}{Style.BRIGHT}System Message:{Style.RESET_ALL} {scc}'{system_message}'{Fore.RESET}")
        print(f"{icc}{Style.BRIGHT}User Input    :{Style.RESET_ALL} {icc}'{input_text}'{Fore.RESET}")
        print(f"{rcc}{Style.BRIGHT}LLM Response  :{Style.RESET_ALL} {rcc}'{response_text.strip()}'{Fore.RESET}\n")
    else:
        if system_message:
            print(f"\n{scc}System Message:{Fore.RESET} '{system_message}'")
        print(f"{icc}User Input    :{Fore.RESET} '{input_text}'")
        print(f"{rcc}LLM Response  :{Fore.RESET} '{response_text.strip()}'\n")


def load_txt_file(f_path):
    """ Load a text file and return the lines as a list """
    with open(f_path, 'r') as file:
        lines = file.readlines()
    return lines


def remove_api_keys(keys_to_remove=None):
    """ Remove the API keys from the environment variables """
    if keys_to_remove is None:
        try:
            keys_to_remove = [x.split("=")[0] for x in load_txt_file(".env")]
            print("Removing all environment variables loaded from .env file")
        except FileNotFoundError:
            print("No .env file found in current directory. No environment variables removed!")

    # Remove all environment variables loaded from .env file
    for key in keys_to_remove:
        os.environ.pop(key)


def retrieve_local_api_keys():
    """ Load the API keys from the .env file using dotenv """
    load_dotenv()
