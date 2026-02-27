import gradio as gr
from pathlib import Path
from ingestion import ingest_document

def process_document_upload(file):
    if file is None:
        return "Upload a file first"

    # Make sure the file from gradio come as a string. if not, get the first element of the list.
    path = file if isinstance(file, str) else file[0]
    filename = Path(path).name
    try:
        summary = ingest_document(file_path=path, filename=filename)
        return f"Summary: {summary}"
    except Exception as e:
        return f"Error processing file: {e}"


# Create a gradio interface with a custom layout using Blocks
with gr.Blocks(title="RAG Doc Chat") as demo:
    # Heading at the top of the page
    gr.Markdown("# RAG-DOC-CHAT")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Upload & Summary")
            # file input is the component whole value (the path) is used as input when the button is clicked
            # setting type to filepath to tell gradio that it should pass the path to the uploaded file as a string to the function, not the file bytes
            file_input = gr.File(label="Upload Document", file_types=[".pdf", ".csv", ".txt", ".md"], type="filepath")
            process_btn = gr.Button("Process")
            # A markdown area where the summary will be displayed
            # whatever the process document upload function returnes, will be displayed in this area
            summary_out = gr.Markdown(label="Document Summary")
        with gr.Column():
            gr.Markdown("## Chat (Phase 5)")
            gr.Markdown("Ask questions about your documents here.")

    # wires the process button to logic
    process_btn.click(fn=process_document_upload, inputs=file_input, outputs=summary_out)

demo.launch()