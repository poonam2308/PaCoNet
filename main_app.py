# main_app.py
import gradio as gr
from app import demo as synthetic_app  # From app.py
from real_image_app import demo as real_app  # From real_image_app.py

with gr.Blocks(title="PaCoNet Combined App") as master_demo:
    gr.Markdown("# 🔀 PaCoNet")

    app_selector = gr.Radio(
        choices=["Synthetic Plot", "Real-World Plot"],
        label="Select Workflow",
        value="Synthetic Plot"
    )

    synthetic_container = gr.Column(visible=False)
    real_container = gr.Column(visible=False)

    with synthetic_container:
        synthetic_app.render()  # Display the imported synthetic demo

    with real_container:
        real_app.render() # Display the imported real-world demo

    def toggle_app(app_choice):
        return (
            gr.update(visible=(app_choice == "Synthetic Plot ")),
            gr.update(visible=(app_choice == "Real-World Plot)"))
        )

    app_selector.change(
        fn=toggle_app,
        inputs=[app_selector],
        outputs=[synthetic_container, real_container]
    )

if __name__ == "__main__":
    master_demo.launch()
