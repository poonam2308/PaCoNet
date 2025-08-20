import gradio as gr
from gr_app.cropping_tab import run_cropping, show_crop
from gr_app.upload_tab import process_upload

def build_ui():
    with gr.Blocks(title="PaCoNet - Data Extraction and Plot Redesign") as demo:
        gr.Markdown("## 📊 PaCoNet - Data Extraction and Plot Redesign")

        # shared state to keep track of upload directory
        upload_dir_state = gr.State()
        crop_dir_state = gr.State()

        with gr.Tabs():
            # --- TAB 1: Upload ---
            with gr.TabItem("1️⃣ Upload"):
                with gr.Row():
                    image_input = gr.Image(type="filepath", label="Upload Image")
                    json_input = gr.File(type="filepath", file_types=[".json"],
                                         label="Upload Annotation (JSON)")
                folder_input = gr.File(
                    type="filepath",
                    file_types=[".zip", ".png", ".jpg", ".jpeg", ".json"],
                    label="Upload Folder (zip or multiple files)",
                    file_count="multiple"
                )

                upload_status = gr.Textbox(label="Result")

            # --- TAB 2: Cropping ---
            with gr.TabItem("2️⃣ Cropping"):
                status = gr.Textbox(label="Status")
                crop_list = gr.Dropdown(label="Select a crop to preview", choices=[])
                preview = gr.Image(label="Crop Preview")

        # hook logic
        upload_btn = gr.Button("Process Upload")
        upload_btn.click(
            fn=process_upload,
            inputs=[image_input, json_input, folder_input],
            outputs=[upload_status, upload_dir_state]   # <-- save dir in session state
        )

        crop_btn = gr.Button("Run Cropping")
        crop_btn.click(
            fn=run_cropping,
            inputs=[upload_dir_state],
            outputs=[status, crop_list, crop_dir_state]
        )

        crop_list.change(
            fn=show_crop,
            inputs=[crop_dir_state, crop_list],
            outputs=preview
        )

    return demo
