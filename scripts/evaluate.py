"""
------------------------------------------------------------
Project:
    RAG-Based Question Answering System for the
    Mesopotamian Civilization

File:
    evaluate.py

Author:
    Deniz Gözcü and Emir Yalçınkaya

Description:
    Evaluates the Mesopotamia RAG system using predefined
    historical questions.

    The evaluation measures:
        - Top-k retrieval accuracy
        - Context relevance using similarity scores
        - Retrieval response time
        - Total RAG response time
        - Answer quality before and after RAG

    Detailed results are displayed in the terminal and saved
    as a JSON file for later use in the project report and
    presentation.

Course:
    Machine Learning and Smart Systems
------------------------------------------------------------
"""


# ----------------------------------------------------------
# Imports
# ----------------------------------------------------------

# json is used to save the final evaluation results
# in a structured and readable format.
import json

# os is used to create file paths that work across
# different operating systems.
import os

# time is used to measure retrieval and response times.
import time

# NumPy is used to prepare the question embedding before
# passing it to the FAISS index.
import numpy as np

# Existing project functions for loading the vector index,
# metadata, and embedding model.
from rag import (
    load_faiss_index,
    load_metadata,
    load_embedding_model,
    build_context,
    build_prompt
)

# Existing function used to send prompts to the LLM.
from llm import generate_answer


# ----------------------------------------------------------
# File and directory constants
# ----------------------------------------------------------

# Folder where generated data and evaluation files are stored.
DATA_DIRECTORY = "data"

# Name of the JSON file that will contain all evaluation results.
RESULTS_FILE_NAME = "evaluation_results.json"

# Complete output path for the evaluation results.
RESULTS_PATH = os.path.join(
    DATA_DIRECTORY,
    RESULTS_FILE_NAME
)


# ----------------------------------------------------------
# Evaluation configuration
# ----------------------------------------------------------

# Number of chunks retrieved for every question.
#
# The project brief recommends retrieving approximately
# three to five relevant chunks. We use three because the
# document is relatively small and structured by topic.
TOP_K = 3

# Controls whether the script should call the LLM twice for
# every question:
#
# 1. Without RAG context
# 2. With retrieved RAG context
#
# Set this to False when testing only retrieval metrics,
# because LLM calls may take more time or use API credits.
RUN_LLM_COMPARISON = True


# ----------------------------------------------------------
# Test questions
# ----------------------------------------------------------

# Each test case contains:
#
# question:
#     The historical question given to the RAG system.
#
# expected_sections:
#     One or more section-title keywords that would count
#     as a correct retrieval.
#
# Multiple expected sections can be provided because some
# questions may be correctly answered from more than one
# chapter of the document.
TEST_CASES = [
    {
        "question": "Why is Mesopotamia called the cradle of civilization?",
        "expected_sections": ["Introduction to Mesopotamia"]
    },
    {
        "question": "What was cuneiform used for?",
        "expected_sections": ["Cuneiform Writing", "The Sumerians"]
    },
    {
        "question": "Who was Hammurabi and why was he important?",
        "expected_sections": ["Babylonian Empire and Hammurabi", "Government and Law"]
    },
    {
        "question": "How did irrigation support Mesopotamian agriculture?",
        "expected_sections": ["Agriculture and Irrigation", "Geography and the Fertile Crescent"]
    },
    {
        "question": "What military advantages did the Assyrians have?",
        "expected_sections": ["Assyrian Empire"]
    },
    {
        "question": "What purpose did ziggurats serve in Mesopotamian civilization?",
        "expected_sections": ["Architecture and Ziggurats", "Religion and Mythology"]
    },
    {
        "question": "How did Mesopotamians contribute to mathematics and astronomy?",
        "expected_sections": ["Science, Mathematics, and Astronomy"]
    },
    {
    "question": "What was Hammurabi's favorite food?",
    "expected_sections": []
    },
    {
        "question": "What themes are explored in the Epic of Gilgamesh?",
        "expected_sections": ["Religion and Mythology"]
    }
]


# ----------------------------------------------------------
# Retrieval functions
# ----------------------------------------------------------

def retrieve_chunks_with_scores(
    question,
    model,
    index,
    metadata,
    top_k=TOP_K
):
    """
    Retrieves the most relevant chunks and their similarity
    scores for one historical question.

    The question is converted into an embedding using the
    same Sentence Transformer model that was used when the
    document chunks were indexed

    Because the stored embeddings and question embedding are
    normalized, an inner-product FAISS index produces scores
    comparable to cosine similarity. A higher score normally
    means stronger semantic similarity

    Parameters
    ----------
    question : str
        Historical question entered into the RAG system.

    model : SentenceTransformer Loaded embedding model.

    index : faiss.Index Loaded FAISS vector index.

    metadata : list
        Metadata corresponding to the indexed text chunks.

    top_k : int
        Number of relevant chunks to retrieve.

    Returns
    -------
    list
        Retrieved chunk dictionaries with similarity scores.
    """

    # Convert the question into a one-row embedding matrix
    #
    # FAISS expects a two-dimensional input shape:
    # number of queries × embedding dimensions
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # FAISS normally expects float32 vectors.
    question_embedding = question_embedding.astype(
        np.float32
    )

    # Search the vector index
    #
    # scores contains the similarity values
    # indices contains the matching metadata positions
    scores, indices = index.search(
        question_embedding,
        top_k
    )

    retrieved_chunks = []

    # scores[0] and indices[0] are used because this script
    # searches only one question at a time.
    for score, metadata_index in zip(
        scores[0],
        indices[0]
    ):
        # FAISS may return -1 when there are fewer available
        # results than the requested top-k value.
        if metadata_index < 0:
            continue

        # Create a copy so the original metadata is not changed.
        chunk = metadata[metadata_index].copy()

        # Convert NumPy float values into normal Python floats
        # so they can be saved correctly as JSON.
        chunk["similarity_score"] = float(score)

        retrieved_chunks.append(chunk)

    return retrieved_chunks


def is_retrieval_correct(
    retrieved_chunks,
    expected_sections
):
    """
    Checks whether at least one expected section appears
    in the top-k retrieved chunks.

    This produces top-k retrieval accuracy.

    Parameters
    ----------
    retrieved_chunks : list
        Chunks returned by the FAISS search.

    expected_sections : list
        Section-title keywords that count as correct.

    Returns
    -------
    bool
        True when an expected section is retrieved.
    """

    # Convert expected titles to lowercase so the comparison
    # is not affected by capitalization.
    expected_lower = [
        section.lower()
        for section in expected_sections
    ]

    for chunk in retrieved_chunks:
        retrieved_title = chunk[
            "section_title"
        ].lower()

        for expected_title in expected_lower:
            # Substring matching is used because the PDF title
            # may contain a number, such as:
            #
            # "5. The Babylonian Empire and Hammurabi"
            if expected_title in retrieved_title:
                return True

    return False


def calculate_average_similarity(retrieved_chunks):
    """
    Calculates the average similarity score of the
    retrieved chunks

    Parameters
    ----------
    retrieved_chunks : list
        Retrieved chunks containing similarity scores.

    Returns
    -------
    float
        Average similarity score, or zero when no chunks
        were retrieved.
    """

    if not retrieved_chunks:
        return 0.0

    scores = [
        chunk["similarity_score"]
        for chunk in retrieved_chunks
    ]

    return sum(scores) / len(scores)


# ----------------------------------------------------------
# Before-and-after RAG comparison
# ----------------------------------------------------------

def create_without_rag_prompt(question):
    """
    Creates a normal LLM prompt without retrieved document
    context.

    This represents the 'before RAG' condition.

    Parameters
    ----------
    question : str
        Historical question.

    Returns
    -------
    str
        Prompt containing only the question.
    """

    return f"""
Answer the following historical question clearly and concisely.

Question:
{question}

Answer:
""".strip()


def generate_comparison_answers(
    question,
    retrieved_chunks
):
    """
    Generates one answer without RAG and one answer with RAG.

    Parameters
    ----------
    question : str
        Historical question.

    retrieved_chunks : list
        Relevant document chunks retrieved by FAISS.

    Returns
    -------
    dict
        Both answers and their measured response times.
    """

    # ------------------------------------------------------
    # Answer without RAG
    # ------------------------------------------------------

    without_rag_prompt = create_without_rag_prompt(
        question
    )

    without_rag_start = time.perf_counter()

    without_rag_answer = generate_answer(
        without_rag_prompt
    )

    without_rag_end = time.perf_counter()

    without_rag_time = (
        without_rag_end - without_rag_start
    )

    # ------------------------------------------------------
    # Answer with RAG
    # ------------------------------------------------------

    # Combine the retrieved document chunks into one context
    context = build_context(retrieved_chunks)

    # Build the grounded prompt that instructs the LLM to
    # answer only from the retrieved document information.
    with_rag_prompt = build_prompt(
        question,
        context
    )

    with_rag_start = time.perf_counter()

    with_rag_answer = generate_answer(
        with_rag_prompt
    )

    with_rag_end = time.perf_counter()

    with_rag_time = with_rag_end - with_rag_start

    return {
        "without_rag_answer": without_rag_answer,
        "without_rag_response_time_seconds": (
            without_rag_time
        ),
        "with_rag_answer": with_rag_answer,
        "with_rag_response_time_seconds": (
            with_rag_time
        )
    }


# ----------------------------------------------------------
# Evaluation process
# ----------------------------------------------------------

def evaluate_test_case(
    test_case,
    model,
    index,
    metadata
):
    """
    Evaluates one test question.

    The function measures retrieval time, checks whether
    the expected section was found, calculates similarity
    scores, and optionally performs the before-and-after
    LLM comparison.

    Parameters
    ----------
    test_case : dict
        Question and expected section information.

    model : SentenceTransformer
        Loaded embedding model.

    index : faiss.Index
        Loaded FAISS vector index.

    metadata : list
        Metadata corresponding to indexed chunks.

    Returns
    -------
    dict
        Complete evaluation result for one question.
    """

    question = test_case["question"]
    expected_sections = test_case[
        "expected_sections"
    ]

    # Measure question embedding and FAISS search together.
    retrieval_start = time.perf_counter()

    retrieved_chunks = retrieve_chunks_with_scores(
        question=question,
        model=model,
        index=index,
        metadata=metadata,
        top_k=TOP_K
    )

    retrieval_end = time.perf_counter()

    retrieval_time = retrieval_end - retrieval_start

    retrieval_correct = is_retrieval_correct(
        retrieved_chunks,
        expected_sections
    )

    average_similarity = calculate_average_similarity(
        retrieved_chunks
    )

    result = {
        "question": question,
        "expected_sections": expected_sections,
        "retrieval_correct": retrieval_correct,
        "retrieval_time_seconds": retrieval_time,
        "average_similarity_score": (
            average_similarity
        ),
        "retrieved_chunks": retrieved_chunks
    }

    # LLM comparison is optional because it creates two API
    # calls for every evaluation question
    if RUN_LLM_COMPARISON:
        comparison = generate_comparison_answers(
            question,
            retrieved_chunks
        )

        result.update(comparison)

        # Total RAG time includes:
        #
        # 1. Question embedding
        # 2. FAISS retrieval
        # 3. LLM answer generation
        result["total_rag_time_seconds"] = (
            retrieval_time
            + comparison[
                "with_rag_response_time_seconds"
            ]
        )

    return result


# ----------------------------------------------------------
# Summary calculations
# ----------------------------------------------------------

def calculate_summary(results):
    """
    Calculates overall evaluation statistics

    Parameters
    ----------
    results : list
        Evaluation results for all questions

    Returns
    -------
    dict
        Summary metrics
    """

    total_questions = len(results)

    # Prevent division by zero if the test list is empty.
    if total_questions == 0:
        return {
            "total_questions": 0,
            "correct_retrievals": 0,
            "top_k_retrieval_accuracy": 0.0,
            "average_retrieval_time_seconds": 0.0,
            "average_similarity_score": 0.0
        }

    correct_retrievals = sum(
        1
        for result in results
        if result["retrieval_correct"]
    )

    retrieval_accuracy = (
        correct_retrievals / total_questions
    )

    average_retrieval_time = (
        sum(
            result["retrieval_time_seconds"]
            for result in results
        )
        / total_questions
    )

    average_similarity = (
        sum(
            result["average_similarity_score"]
            for result in results
        )
        / total_questions
    )

    summary = {
        "total_questions": total_questions,
        "top_k": TOP_K,
        "correct_retrievals": correct_retrievals,
        "top_k_retrieval_accuracy": retrieval_accuracy,
        "average_retrieval_time_seconds": (
            average_retrieval_time
        ),
        "average_similarity_score": (
            average_similarity
        )
    }

    if RUN_LLM_COMPARISON:
        average_without_rag_time = (
            sum(
                result[
                    "without_rag_response_time_seconds"
                ]
                for result in results
            )
            / total_questions
        )

        average_with_rag_time = (
            sum(
                result[
                    "with_rag_response_time_seconds"
                ]
                for result in results
            )
            / total_questions
        )

        average_total_rag_time = (
            sum(
                result["total_rag_time_seconds"]
                for result in results
            )
            / total_questions
        )

        summary[
            "average_without_rag_llm_time_seconds"
        ] = average_without_rag_time

        summary[
            "average_with_rag_llm_time_seconds"
        ] = average_with_rag_time

        summary[
            "average_total_rag_time_seconds"
        ] = average_total_rag_time

    return summary


# ----------------------------------------------------------
# Output functions
# ----------------------------------------------------------

def print_test_result(result, test_number):
    """
    Displays one evaluation result in a readable format.

    Parameters
    ----------
    result : dict
        Evaluation result for one question.

    test_number : int
        Number of the current test.
    """

    print("\n" + "=" * 70)
    print(f"TEST {test_number}")
    print("=" * 70)

    print(f"\nQuestion:\n{result['question']}")

    print("\nExpected section(s):")

    for section in result["expected_sections"]:
        print(f"- {section}")

    print("\nRetrieved chunks:")

    for rank, chunk in enumerate(
        result["retrieved_chunks"],
        start=1
    ):
        print(
            f"\n{rank}. {chunk['section_title']}"
        )
        print(
            f"   Page: {chunk['source_page']}"
        )
        print(
            "   Similarity score: "
            f"{chunk['similarity_score']:.4f}"
        )

    status = (
        "CORRECT"
        if result["retrieval_correct"]
        else "INCORRECT"
    )

    print(f"\nRetrieval result: {status}")

    print(
        "Retrieval time: "
        f"{result['retrieval_time_seconds']:.4f} seconds"
    )

    print(
        "Average context similarity: "
        f"{result['average_similarity_score']:.4f}"
    )

    if RUN_LLM_COMPARISON:
        print("\n" + "-" * 70)
        print("ANSWER WITHOUT RAG")
        print("-" * 70)
        print(result["without_rag_answer"])

        print(
            "\nResponse time without RAG: "
            f"{result['without_rag_response_time_seconds']:.4f} "
            "seconds"
        )

        print("\n" + "-" * 70)
        print("ANSWER WITH RAG")
        print("-" * 70)
        print(result["with_rag_answer"])

        print(
            "\nResponse time with RAG: "
            f"{result['with_rag_response_time_seconds']:.4f} "
            "seconds"
        )

        print(
            "Total RAG time: "
            f"{result['total_rag_time_seconds']:.4f} "
            "seconds"
        )


def print_summary(summary):
    """
    Displays the overall evaluation metrics.

    Parameters
    ----------
    summary : dict
        Calculated evaluation summary.
    """

    print("\n\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        "\nQuestions tested: "
        f"{summary['total_questions']}"
    )

    print(
        "Correct top-k retrievals: "
        f"{summary['correct_retrievals']}"
    )

    accuracy_percentage = (
        summary["top_k_retrieval_accuracy"] * 100
    )

    print(
        f"Top-{summary['top_k']} retrieval accuracy: "
        f"{accuracy_percentage:.2f}%"
    )

    print(
        "Average similarity score: "
        f"{summary['average_similarity_score']:.4f}"
    )

    print(
        "Average retrieval time: "
        f"{summary['average_retrieval_time_seconds']:.4f} "
        "seconds"
    )

    if RUN_LLM_COMPARISON:
        print(
            "Average LLM time without RAG: "
            f"{summary['average_without_rag_llm_time_seconds']:.4f} "
            "seconds"
        )

        print(
            "Average LLM time with RAG: "
            f"{summary['average_with_rag_llm_time_seconds']:.4f} "
            "seconds"
        )

        print(
            "Average total RAG response time: "
            f"{summary['average_total_rag_time_seconds']:.4f} "
            "seconds"
        )


def save_results(results, summary):
    """
    Saves detailed results and summary metrics as JSON.

    Parameters
    ----------
    results : list
        Detailed results for every question.

    summary : dict
        Overall evaluation statistics.
    """

    evaluation_output = {
        "configuration": {
            "top_k": TOP_K,
            "llm_comparison_enabled": (
                RUN_LLM_COMPARISON
            )
        },
        "summary": summary,
        "test_results": results
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            evaluation_output,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nEvaluation results saved to:\n{RESULTS_PATH}"
    )


# ----------------------------------------------------------
# Main evaluation pipeline
# ----------------------------------------------------------

def main():
    """
    Runs the complete RAG evaluation process.
    """

    print("=" * 70)
    print("Mesopotamia RAG System Evaluation")
    print("=" * 70)

    print("\nLoading evaluation resources...")

    # Resources are loaded once and reused for every question.
    # This avoids repeatedly loading the embedding model and
    # FAISS index, which would make evaluation much slower.
    index = load_faiss_index()
    metadata = load_metadata()
    model = load_embedding_model()

    print("\nResources loaded successfully.")

    results = []

    for test_number, test_case in enumerate(
        TEST_CASES,
        start=1
    ):
        print(
            f"\nEvaluating question "
            f"{test_number}/{len(TEST_CASES)}..."
        )

        result = evaluate_test_case(
            test_case=test_case,
            model=model,
            index=index,
            metadata=metadata
        )

        results.append(result)

        print_test_result(
            result,
            test_number
        )

    summary = calculate_summary(results)

    print_summary(summary)

    save_results(
        results,
        summary
    )

    print("\nEvaluation completed successfully.")


# ----------------------------------------------------------
# Program Entry Point
# ----------------------------------------------------------

if __name__ == "__main__":
    main()