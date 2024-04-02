from dotenv import load_dotenv
import logging
import sys

load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

## -----------------------------
from llama_index import download_loader

WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()

documents = loader.load_data(pages=['Natural Language Processing', 'Artificial Intelligence'])

from llama_index.node_parser import SimpleNodeParser

# Assuming documents have already been loaded

# Initialize the parser
parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=20)

# Parse documents into nodes
nodes = parser.get_nodes_from_documents(documents)
print(len(nodes))

## now load and save to deeplake data store
from llama_index.vector_stores import DeepLakeVectorStore

my_activeloop_org_id = "hunter"
my_activeloop_dataset_name = "wikiloader"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=False)

from llama_index.storage.storage_context import StorageContext
from llama_index import VectorStoreIndex

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context
)