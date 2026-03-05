import gradio as gr
from pathlib import Path
from ingestion import ingest_document_stream
from query import answer_question
from src.logger import get_logger

logger = get_logger(__name__)

def process_document_upload(file, request: gr.Request):
    if file is None:
        return "Upload a file first"

    # Make sure the file from gradio come as a string. if not, get the first element of the list.
    path = file if isinstance(file, str) else file[0]
    filename = Path(path).name

    logger.info(f"Processing document upload for user {request.username} with file --{filename}--")

    yield 10, "⏳ Loading and processing document..."
    try:
        for progress, text in ingest_document_stream(username=request.username, file_path=path, filename=filename):
            yield progress, text

        logger.info(f"Document upload complete for user {request.username} with file --{filename}--")
    except ValueError as e:
        yield 100, f"Invalid file format: {e}"
        logger.error(f"Invalid file format: {e} for user {request.username} with file --{filename}--")
    except Exception as e:
        yield 100, f"Error processing file: {e}"
        logger.error(f"Error processing file: {e} for user {request.username} with file --{filename}--")

def get_model_answer(history, question, request: gr.Request):
    if not (question or "").strip():
        logger.warning(f"No question provided for user {request.username}")
        return history, ""

    result = answer_question(username=request.username, question=question)
    
    user_msg = {"role": "user", "content": [{"type": "text", "text": question}]}
    assistant_content = result.answer
    if result.sources:
        assistant_content += result.sources
    assistant_msg = {"role": "assistant", "content": [{"type": "text", "text": assistant_content}]}

    logger.info(f"Answer generated for user {request.username} with question --{question}-- and answer --{assistant_content}--")

    new_history = history + [user_msg, assistant_msg]
    return new_history, ""


# Create a gradio interface with a custom layout using Blocks
with gr.Blocks(title="RAG Doc Chat") as demo:
    # Heading at the top of the page
    gr.Markdown("# RAG-DOC-CHAT")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Upload & Summary")
            progress_bar = gr.Progress()
            # file input is the component whole value (the path) is used as input when the button is clicked
            # setting type to filepath to tell gradio that it should pass the path to the uploaded file as a string to the function, not the file bytes
            file_input = gr.File(label="Upload Document", file_types=[".pdf", ".csv", ".txt", ".md"], type="filepath")
            process_btn = gr.Button("Process")
            # A markdown area where the summary will be displayed
            # whatever the process document upload function returnes, will be displayed in this area
            progress_bar = gr.Slider(0, 100, value=0, label="Progress", interactive=False)
            summary_out = gr.Markdown(label="")
        with gr.Column():
            gr.Markdown("## Chat (Phase 5)")
            gr.Markdown("Ask questions about your documents here.")
            chatbot = gr.Chatbot(label="Chat", value=[])
            answer_textbox = gr.Textbox(label="Question", placeholder="What is this document about?", lines=2)
            ask_btn = gr.Button("Ask")
            gr.Markdown("Sources")

    # wires the process button to logic
    process_btn.click(fn=process_document_upload, inputs=file_input, outputs=[progress_bar, summary_out])
    ask_btn.click(fn=get_model_answer, inputs=[chatbot, answer_textbox], outputs=[chatbot, answer_textbox])

demo.queue()
demo.launch(auth=[("admin", "admin"), ("user1", "user1"), ("eran", "eran")])