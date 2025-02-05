import urllib.request
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import HTMLToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from dotenv import load_dotenv

load_dotenv()

# Download sample document
urllib.request.urlretrieve(
    "https://www.oreilly.com/openbook/freedom/ch01.html",
    "free_as_in_freedom.html"
)

# Document Processing Pipeline
document_store = InMemoryDocumentStore()

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", HTMLToDocument())
indexing_pipeline.add_component("cleaner", DocumentCleaner())
indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=5))
indexing_pipeline.add_component("embedder", OpenAIDocumentEmbedder())
indexing_pipeline.add_component("writer", DocumentWriter(document_store))

indexing_pipeline.connect("converter.documents", "cleaner.documents")
indexing_pipeline.connect("cleaner.documents", "splitter.documents")
indexing_pipeline.connect("splitter.documents", "embedder.documents")
indexing_pipeline.connect("embedder.documents", "writer.documents")

# Run indexing pipeline
indexing_result = indexing_pipeline.run(data={"sources": ["free_as_in_freedom.html"]})
print("Indexing complete. Documents written:", indexing_result["writer"]["documents_written"])

# RAG Pipeline
template = """Given these documents, answer the question.
Documents:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}
Question: {{query}}
Answer:"""

rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", OpenAITextEmbedder())
rag_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store))
rag_pipeline.add_component("prompt_builder", PromptBuilder(template=template))
rag_pipeline.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo"))

rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")

# Example queries
queries = [
    "What is the profession of Richard M. Stallman and where does he work?",
    "Why did Stallman get frustrated when he tried to retrieve his print job from the new Xerox printer?"
]

for query in queries:
    result = rag_pipeline.run({
        "text_embedder": {"text": query},
        "prompt_builder": {"query": query}
    })
    print(f"\nQuestion: {query}")
    print("Answer:", result["llm"]["replies"][0])
    print("Usage:", result["llm"]["meta"][0]["usage"])