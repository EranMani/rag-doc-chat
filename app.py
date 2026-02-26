import gradio as gr

def process_document_upload(file):
    if file is None:
        return "Upload a file first"

    return "Summary will go here!"

with gr.Blocks(title="RAG Doc Chat") as demo:
    gr.Markdown("# Upload a document to get started")
    file_input = gr.File(label="Upload Document", file_types=[".pdf", ".docx", ".txt", ".md"])
    process_btn = gr.Button("Process")
    summary_out = gr.Markdown(label="Document Summary")

    process_btn.click(fn=process_document_upload, inputs=file_input, outputs=summary_out)

demo.launch()