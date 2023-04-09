from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import argparse
import pickle
from misc.utils import retrieve_local_api_keys, save_pickle_file

from chat_with_x.utils.processing_utils import DataLoaderMapper


def create_vectorstore_from_local_file(f_path, chunk_size=1024, chunk_overlap=128, no_save=True, output_fpath=None,
                                       **kwargs):
    """
    Create a vectorstore from any input file that contains natural language.
    We must determine the file type and use it to determine the appropriate dataloader

    Args:
        f_path (str):
            – Absolute path to the file to be loaded
        chunk_size (int, optional):
            – The size of the chunks to be used when splitting the text
        chunk_overlap (int, optional):
            – The amount of overlap between chunks
        no_save (bool, optional):
            – Whether to save the vectorstore to disk.
        output_fpath (str, optional):
            – The directory to save the vectorstore to. If None, the vectorstore will be saved at the location
              of the input file as a pickle file i.e. 'x/y/z.txt' -> 'x/y/z_vectorstore.pkl'
        **kwargs:
            – Additional keyword arguments
    Returns:
        The vectorstore object
    """

    # Determine file type and instantiate the appropriate data loader
    mapper = DataLoaderMapper()
    loader = mapper.get_loader_from_path(f_path)

    # Load data
    raw_docs = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = text_splitter.split_documents(raw_docs)

    # Load Data to vectorstore (FAISS) using text-embedding-ada-002 OpenAI model to embed data
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunked_docs, embeddings)

    if output_fpath is None:
        output_fpath = f_path.replace(".txt", "_vectorstore.pkl")

    # Determine output path or simply use passed argument
    if not no_save:
        save_pickle_file(output_fpath, vectorstore)

    return vectorstore, None if no_save else output_fpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a vectorstore from any input file that contains language/text")
    parser.add_argument("f_path", help="Path to the file to be loaded")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Chunk size to be used when splitting the text")
    parser.add_argument("--chunk_overlap", type=int, default=128, help="The amount of overlap between chunks")
    parser.add_argument("--output_fpath", help="The directory to save the vectorstore to. If not specified, "
                                               "the vectorstore will be saved at the location of the input file as a "
                                               "pickle file")
    parser.add_argument("--no_save", action="store_true", help="Whether to save the vectorstore to disk.")

    args = parser.parse_args()

    # Authenticate
    try:
        retrieve_local_api_keys()
    except:
        raise Exception("Could not authenticate. Please ensure that you have a file named '.env' in the root "
                        "directory of this project. This file should contain your OpenAI API key. We are utilizing "
                        "the `python-dotenv` library in this project. See the README for more details.")

    # Create vectorstore
    _, vs_path = create_vectorstore_from_local_file(**args.__dict__)
    print(f"\n\n... Vectorstore successfully saved to: {vs_path} ...\n")
