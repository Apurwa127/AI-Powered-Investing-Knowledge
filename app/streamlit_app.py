import sys
from pathlib import Path

# Locate the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root to Python's import path
sys.path.insert(0, str(PROJECT_ROOT))

# Debug information
print("Project root:", PROJECT_ROOT)
print("App folder exists:", (PROJECT_ROOT / "app").is_dir())
print("App package exists:", (PROJECT_ROOT / "app" / "__init__.py").is_file())






import streamlit as st

from app.embedding_service import EmbeddingService
from app.retrieval_service import RetrievalService
from app.llm_service import LLMService
from app.reranker import Reranker
from app.rag_service import RAGService


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Investing Knowledge",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Initialize services
# --------------------------------------------------

@st.cache_resource
def initialize_rag_service():

    # Initialize embedding service
    embedding_service = EmbeddingService()

    # Initialize retrieval service
    retrieval_service = RetrievalService(
        embedding_service=embedding_service
    )

    # Initialize LLM service
    llm_service = LLMService()

    # Initialize reranker
    reranker = Reranker()

    # Initialize RAG service
    rag_service = RAGService(
        retrieval_service=retrieval_service,
        llm_service=llm_service,
        reranker=reranker
    )

    return rag_service


# --------------------------------------------------
# Application interface
# --------------------------------------------------

st.title("📊 AI-Powered Investing Knowledge")

st.write(
    "An AI-powered financial research assistant "
    "that answers questions using company annual reports."
)

st.divider()


# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------

st.sidebar.header("Search Filters")

company = st.sidebar.selectbox(
    "Select Company",
    ["All Companies", "Apple", "Microsoft", "Tesla"]
)

fiscal_year = st.sidebar.selectbox(
    "Fiscal Year",
    ["All Years", 2024]
)

# Convert selections into retrieval filters
selected_company = (
    None if company == "All Companies" else company
)

selected_year = (
    None if fiscal_year == "All Years" else fiscal_year
)


# --------------------------------------------------
# Question input
# --------------------------------------------------

st.subheader("Ask a Financial Question")

with st.form("question_form"):

    question = st.text_area(
        "Enter your question",
        placeholder=(
            "Example: What was Apple's total revenue in 2024?"
        ),
        height=100
    )

    submit_button = st.form_submit_button(
        "Generate Answer",
        type="primary"
    )


# --------------------------------------------------
# Process question
# --------------------------------------------------

if submit_button:

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        try:

            with st.spinner(
                "Searching financial reports and generating answer..."
            ):

                # Initialize RAG pipeline
                rag_service = initialize_rag_service()

                # Generate answer
                result = rag_service.ask_question(
                    question=question,
                    company=selected_company,
                    fiscal_year=selected_year,
                    top_k=3,
                    retrieval_k=10
                )

            # --------------------------------------
            # Display answer
            # --------------------------------------

            st.divider()

            st.subheader("Answer")

            st.markdown(result["answer"])

            # --------------------------------------
            # Display sources
            # --------------------------------------

            st.subheader("Sources")

            if result["sources"]:

                for index, source in enumerate(
                    result["sources"], start=1
                ):

                    with st.expander(
                        f"Source {index}: {source['company']} "
                        f"({source['fiscal_year']})"
                    ):

                        st.write(
                            "**Document:**",
                            source["source"]
                        )

                        st.write(
                            "**Page:**",
                            source["page_number"]
                        )

                        st.write(
                            "**Reranking score:**",
                            round(source["rerank_score"], 4)
                        )

            else:

                st.info("No source documents were retrieved.")

            # --------------------------------------
            # Citation validation
            # --------------------------------------

            if result["invalid_citations"]:

                st.warning(
                    "Some citations could not be matched "
                    "to the retrieved source documents."
                )

                for citation in result["invalid_citations"]:
                    st.write(citation)

        except Exception as error:

            st.error(
                f"An error occurred while processing your question: {error}"
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "AI-Powered Investing Knowledge | "
    "Powered by Azure AI Search, OpenAI, and LangChain components"
)
