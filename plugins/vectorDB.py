import os
import uuid
import chromadb
from chromadb.config import Settings
from plugin_base import Plugin

PERSIST_DIR = "./chroma_data"

class ChromaManager:
    """
    Manages a single persistent ChromaDB collection named 'a0_docs'.
    All plugins share this collection. We persist to PERSIST_DIR so
    data survives restarts.
    """
    _client = None
    _collection = None

    @classmethod
    def get_collection(cls):
        """Lazy-init the Chroma client and collection, then return it."""
        if cls._client is None:
            cls._client = chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=PERSIST_DIR
                )
            )
        if cls._collection is None:
            cls._collection = cls._client.get_or_create_collection(name="a0_docs")
        return cls._collection

    @classmethod
    def persist(cls):
        """
        With 'persist_directory' set in Settings, changes auto-save. 
        We provide this hook in case you want to do more advanced handling.
        """
        pass

    @classmethod
    def clear_db(cls):
        """Delete the 'a0_docs' collection entirely."""
        if cls._client is not None:
            try:
                cls._client.delete_collection(name="a0_docs")
            except:
                pass
            cls._collection = None


class DBStorePlugin(Plugin):
    """
    -dbstore <text to store>

    Stores <text> in the persistent Chroma DB.
    """
    @property
    def command(self) -> str:
        return "-dbstore"

    def run(self, user_input: str, chat_history: list, copyClipboard) -> str or None:
        parts = user_input.split(" ", 1)
        if len(parts) < 2:
            print("Usage: -dbstore <text>")
            return None

        text_to_store = parts[1].strip()
        coll = ChromaManager.get_collection()

        doc_id = str(uuid.uuid4())
        coll.add(
            documents=[text_to_store],
            metadatas=[{"source": "manual"}],
            ids=[doc_id]
        )
        ChromaManager.persist()

        print(f"Stored doc ID={doc_id}")
        return None  # No GPT call


class DBQueryPlugin(Plugin):
    """
    -dbquery <your question>

    Looks up the nearest doc in the DB, returns a combined prompt so
    the assistant sees both your question and the relevant doc.
    """
    @property
    def command(self) -> str:
        return "-dbquery"

    def run(self, user_input: str, chat_history: list, copyClipboard) -> str or None:
        parts = user_input.split(" ", 1)
        if len(parts) < 2:
            print("Usage: -dbquery <question>")
            return None

        query = parts[1].strip()
        coll = ChromaManager.get_collection()

        # If no docs exist, just exit
        if coll.count() == 0:
            print("No documents found in the DB!")
            return None

        # Query for the single top match
        results = coll.query(
            query_texts=[query],
            n_results=1
        )
        best_doc = results["documents"][0][0]
        best_id = results["ids"][0][0]
        print(f"Top doc ID={best_id}, snippet:\n{best_doc[:200]}...")

        # Return a new user prompt that merges the user question + best doc
        new_input = (
            f"{query}\n\n"
            f"-----\n"
            f"Relevant doc:\n"
            f"{best_doc}\n"
            f"-----\n"
        )
        return new_input


class DBClearPlugin(Plugin):
    """
    -dbclear

    Deletes the entire 'a0_docs' collection. Everything is gone.
    """
    @property
    def command(self) -> str:
        return "-dbclear"

    def run(self, user_input: str, chat_history: list, copyClipboard) -> str or None:
        ChromaManager.clear_db()
        print("Chroma database cleared!")
        return None


class DBEmbedPlugin(Plugin):
    """
    -dbembed

    Reads the file 'embed.txt' (assumed in the same folder as main.py)
    and inserts its content as a single doc in Chroma.
    """
    @property
    def command(self) -> str:
        return "-dbembed"

    def run(self, user_input: str, chat_history: list, copyClipboard) -> str or None:
        embed_file = "embed.txt"
        if not os.path.isfile(embed_file):
            print("No embed.txt found!")
            return None

        with open(embed_file, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            print("embed.txt is empty!")
            return None

        coll = ChromaManager.get_collection()
        doc_id = str(uuid.uuid4())
        coll.add(
            documents=[content],
            metadatas=[{"source": "embed_file"}],
            ids=[doc_id]
        )
        ChromaManager.persist()

        print(f"Embedded content from embed.txt => doc ID={doc_id}")
        return None
