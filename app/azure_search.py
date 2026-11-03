
import os

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile
)


load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")

AZURE_SEARCH_INDEX_NAME = "investing-knowledge-index"


def create_search_index():

    if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_API_KEY:
        raise ValueError("Azure AI Search credentials are missing")

    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
    )

    fields = [

        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True
        ),

        SearchableField(
            name="content",
            type=SearchFieldDataType.String
        ),

        SearchableField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True
        ),

        SimpleField(
            name="document_id",
            type=SearchFieldDataType.String,
            filterable=True
        ),

        SearchableField(
            name="company",
            type=SearchFieldDataType.String,
            filterable=True
        ),

        SimpleField(
            name="fiscal_year",
            type=SearchFieldDataType.Int32,
            filterable=True
        ),

        SimpleField(
            name="page_number",
            type=SearchFieldDataType.Int32,
            filterable=True
        ),

        SimpleField(
            name="document_type",
            type=SearchFieldDataType.String,
            filterable=True
        ),

        SimpleField(
            name="chunk_index",
            type=SearchFieldDataType.Int32,
            filterable=True
        ),

        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(
                SearchFieldDataType.Single
            ),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="my-vector-profile"
        )
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="my-hnsw-config"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="my-vector-profile",
                algorithm_configuration_name="my-hnsw-config"
            )
        ]
    )

    index = SearchIndex(
        name=AZURE_SEARCH_INDEX_NAME,
        fields=fields,
        vector_search=vector_search
    )

    result = index_client.create_or_update_index(index)

    print(f"Index successfully created or updated: {result.name}")

    return result


if __name__ == "__main__":
    create_search_index()
