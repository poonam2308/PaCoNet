import gradio as gr

# --- Import your processing modules ---
from .detection_tab import process_input, update_coordinate_selector, update_coords_for_image, save_selected_axes, \
    toggle_params
from .cropping_tab import crop_and_return_images, get_base_overlay_with_axes, generate_cropping_overlay, \
    save_selected_images, select_crop_from_gallery
from .separation_tab import run_category_separation, generate_category_overlay, select_category_from_gallery
from .denoising_tab import run_denoising, generate_denoised_overlay, select_denoised_from_gallery
from .prediction_tab import trigger_line_prediction_all, generate_predicted_overlay_pre, \
    generate_predicted_overlay_post, select_prediction_from_gallery_pre, select_prediction_from_gallery_post, \
    generate_predicted_overlay_mask, generate_predicted_overlay_mask_post, select_prediction_from_gallery_mask, \
    select_prediction_from_gallery_mask_post
from .stitching_tab import run_stitching_from_prediction
from .session import save_session_log, SESSION


def build_ui():
    with gr.Blocks(title="PaCoNet - Data Extraction and Plot Redesign") as demo:
        gr.Markdown("## 📊 PaCoNet - Data Extraction and Plot Redesign")

        with gr.Tabs():
            # --- TAB 1: Line Detection ---
            with gr.TabItem("1️⃣ Detection"):
                with gr.Row():
                    file_input = gr.File(file_types=[".png", ".jpg", ".jpeg"], file_count="multiple",
                                         label="Upload Image(s)")
                    json_input = gr.File(file_types=[".json"], file_count="single",
                                         label="Upload Metadata JSON (optional)")

                with gr.Group(visible=True) as param_group:
                    aperture = gr.Slider(3, 7, step=2, value=5, label="Canny Aperture Size")
                    min_len = gr.Slider(10, 150, step=2, value=20, label="Min Line Length")
                    max_gap = gr.Slider(1, 50, step=1, value=1, label="Max Line Gap")
                    min_spacing_slider = gr.Slider(5, 100, step=1, value=20, label="Minimum Spacing Between Axes (px)")
                    left_thresh_slider = gr.Slider(0.0, 0.2, step=0.01, value=0.03,
                                                   label="Left Edge Ignore Threshold (fraction)")
                    right_thresh_slider = gr.Slider(0.8, 1.0, step=0.01, value=0.95,
                                                    label="Right Edge Ignore Threshold (fraction)")

                json_input.change(toggle_params, inputs=json_input, outputs=param_group)
                run_btn = gr.Button("Run Detection")
                img_output = gr.Gallery(label="Detected Images with Lines", columns=[3], height=400)
                # with gr.Row():
                #     image_selector = gr.Dropdown(choices=[], label="Select Image for Coordinate Filtering")
                #     coord_selector = gr.CheckboxGroup(choices=[], label="Select X-Coordinates to Keep")
                #     save_coords_btn = gr.Button("💾 Save Selected Coordinates")
                #     save_coords_status = gr.Textbox(label="Save Status", interactive=False)

            # --- TAB 2: Cropping ---
            with gr.TabItem("2️⃣ Cropping") as cropping_tab:
                # original_input_img = gr.Image(label="Original Image", type="filepath")
                crop_btn = gr.Button("Crop Images")
                cropped_output = gr.Gallery(label="Cropped Images", columns=[4], height=300)
                cropped_selection = gr.CheckboxGroup(label="Select Cropped Files to Save", choices=[], visible=False)
                save_selected_btn = gr.Button("Save Selected Crops", visible=False)
                save_result_text = gr.Textbox(label="Save Result", interactive=False, visible=False)

                # 🔍 New overlay visualization in cropping tab
                crop_overlay_img = gr.Image(label="Overlay with Crop Highlight", type="filepath")

            # --- TAB 3: Category Separation ---
            with gr.TabItem("3️⃣ Category Separation") as sep_tab:
                method_dropdown = gr.Dropdown(
                    choices=["peaks", "topk"],
                    value="topk",
                    label="Hue Separation Method"
                )
                topk_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=3,
                    label="Top K Dominant Hues (Only used if method='topk')"
                )

                sep_btn = gr.Button("Run Category Separation", visible=False)
                sep_result_text = gr.Textbox(label="Separation Status", interactive=False, visible=False)
                sep_gallery = gr.Gallery(label="Separated Categories", columns=[4], visible=False)
                denoise_selection = gr.CheckboxGroup(label="Select Images for Denoising", choices=["ALL"],
                                                     visible=False)
                category_overlay_img = gr.Image(label="Overlay: Selected Category Highlight", type="filepath",
                                                visible=True)

            # --- TAB 4: Denoising ---
            with gr.TabItem("4️⃣ Denoising"):
                run_denoise_btn = gr.Button("Run UNet Denoising", visible=False)
                denoise_gallery = gr.Gallery(label="Denoised Images", columns=[4], visible=False)
                denoise_status = gr.Textbox(label="Denoising Status", interactive=False, visible=False)
                denoised_selection = gr.CheckboxGroup(label="Select Denoised Overlays", choices=[], visible=False)
                denoise_overlay_img = gr.Image(
                    label="Overlay: Denoised Categories on Original",
                    type="filepath",
                    visible=True
                )

            with gr.TabItem("5️⃣ Line Prediction"):
                line_threshold_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.5,
                    label="Line Confidence Threshold"
                )

                predict_btn = gr.Button("Run Line Prediction")

                # --- NEW: PRE ---
                gr.Markdown("**Pre-masked filtering predictions**")
                svg_gallery_pre = gr.Gallery(label="Pre SVGs", columns=3)
                json_output_pre = gr.Textbox(label="Pre JSON Outputs", lines=10, visible=False)
                pred_overlay_selection_pre = gr.CheckboxGroup(
                    label="Select Pre Predicted Lines to Overlay",
                    choices=[],
                    visible=False
                )
                pred_overlay_img_pre = gr.Image(
                    label="Overlay: Pre Predicted Lines on Original",
                    type="filepath",
                    visible=True
                )

                # --- EXISTING: POST ---
                gr.Markdown("**Post-masked filtering predictions**")
                svg_gallery = gr.Gallery(label="Post SVGs", columns=3)
                json_output = gr.Textbox(label="Post JSON Outputs", lines=10, visible=False)
                pred_overlay_selection = gr.CheckboxGroup(
                    label="Select Post Predicted Lines to Overlay",
                    choices=[],
                    visible=False
                )
                pred_overlay_img = gr.Image(
                    label="Overlay: Post Predicted Lines on Original",
                    type="filepath",
                    visible=True
                )

                # --- NEW: MASK ---
                gr.Markdown("**Mask-filtered predictions**")
                svg_gallery_mask = gr.Gallery(label="Mask SVGs", columns=3)
                json_output_mask = gr.Textbox(label="Mask JSON Outputs", lines=10, visible=False)
                pred_overlay_selection_mask = gr.CheckboxGroup(
                    label="Select Mask Predicted Lines to Overlay", choices=[], visible=False
                )
                pred_overlay_img_mask = gr.Image(label="Overlay: Mask Predicted Lines", type="filepath", visible=True)

                # --- NEW: MASK+POST ---
                gr.Markdown("**Mask + Postprocess predictions**")
                svg_gallery_mask_post = gr.Gallery(label="Mask+Post SVGs", columns=3)
                json_output_mask_post = gr.Textbox(label="Mask+Post JSON Outputs", lines=10, visible=False)
                pred_overlay_selection_mask_post = gr.CheckboxGroup(
                    label="Select Mask+Post Predicted Lines to Overlay", choices=[], visible=False
                )
                pred_overlay_img_mask_post = gr.Image(label="Overlay: Mask+Post Predicted Lines", type="filepath",
                                                      visible=True)

            with gr.TabItem("6️⃣ Stitch & View CSVs"):
                threshold_input = gr.Slider(minimum=1.0, maximum=50.0, step=1.0, value=10.0, label="Matching Threshold")
                use_hsv_checkbox = gr.Checkbox(label="Use Custom HSV Colors", value=True)
                run_stitch_btn = gr.Button("Run Stitching from Predictions")
                csv_list_output = gr.Dataframe(
                    headers=["Stitched CSV Filename", " "],
                    datatype=["str", "str"],
                    interactive=False,
                    label="Stitched CSV Files (per Category)"
                )
                svg_viewer = gr.Image(label="Redesigned Plot", type="filepath")
                csv_status = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            download_log_btn = gr.Button("📥 Download Session Log")
            log_file_output = gr.File(label="Downloadable Session Summary")

        # --- Hook up logic ---
        # run_btn.click(fn=process_input,
        #               inputs=[file_input, json_input, aperture, min_len, max_gap, min_spacing_slider, left_thresh_slider,
        #                       right_thresh_slider],
        #               outputs=[img_output]
        #               ).then(
        #     fn=update_coordinate_selector,
        #     inputs=[],
        #     outputs=[image_selector, coord_selector]
        # )
        #
        # image_selector.change(fn=update_coords_for_image,
        #                       inputs=[image_selector],
        #                       outputs=[coord_selector])
        #
        # save_coords_btn.click(fn=save_selected_axes,
        #                       inputs=[image_selector, coord_selector],
        #                       outputs=[save_coords_status])

        run_btn.click(
            fn=process_input,
            inputs=[file_input, json_input, aperture, min_len, max_gap,
                    min_spacing_slider, left_thresh_slider, right_thresh_slider],
            outputs=[img_output]
        )

        crop_btn.click(
            fn=crop_and_return_images,
            outputs=[
                cropped_output, cropped_selection,
                cropped_selection, save_selected_btn, save_result_text, sep_btn
            ]
        ).then(
            fn=get_base_overlay_with_axes,
            outputs=[crop_overlay_img]
        )

        cropped_selection.change(
            fn=generate_cropping_overlay,
            inputs=[cropped_selection],
            outputs=[crop_overlay_img]
        )

        cropped_output.select(
            fn=select_crop_from_gallery,
            inputs=[],
            outputs=[cropped_selection]
        )

        save_selected_btn.click(fn=save_selected_images,
                                inputs=[cropped_selection],
                                outputs=[save_result_text])

        sep_btn.click(
            fn=run_category_separation,
            inputs=[method_dropdown, topk_slider],
            outputs=[
                sep_gallery,
                sep_gallery,
                sep_result_text,
                denoise_selection,
                run_denoise_btn,
                denoise_status
            ]
        ).then(
            fn=lambda: generate_category_overlay([]),  # 👈 default: blurred only
            outputs=[category_overlay_img]
        )

        sep_gallery.select(
            fn=select_category_from_gallery,
            inputs=[],
            outputs=[denoise_selection]
        )

        denoise_selection.change(
            fn=generate_category_overlay,
            inputs=[denoise_selection],
            outputs=[category_overlay_img]
        ).then(
            fn=lambda: generate_denoised_overlay([]),  # 👈 default blurred only
            outputs=[denoise_overlay_img]
        )

        denoise_gallery.select(
            fn=select_denoised_from_gallery,
            inputs=[],
            outputs=[denoised_selection]
        )

        run_denoise_btn.click(
            fn=run_denoising,
            inputs=[denoise_selection],  # existing category selection
            outputs=[denoise_gallery, denoise_gallery, denoise_status, denoised_selection]
        )

        denoised_selection.change(
            fn=generate_denoised_overlay,
            inputs=[denoised_selection],
            outputs=[denoise_overlay_img]
        )

        predict_btn.click(
            fn=trigger_line_prediction_all,
            inputs=[line_threshold_slider],
            outputs=[
                # PRE
                svg_gallery_pre, json_output_pre, pred_overlay_selection_pre, pred_overlay_img_pre,
                # MASK
                svg_gallery_mask, json_output_mask, pred_overlay_selection_mask, pred_overlay_img_mask,
                # POST
                svg_gallery, json_output, pred_overlay_selection, pred_overlay_img,
                # MASK+POST
                svg_gallery_mask_post, json_output_mask_post, pred_overlay_selection_mask_post,
                pred_overlay_img_mask_post,
            ]
        ).then(
            fn=lambda: generate_predicted_overlay_pre([]),
            outputs=[pred_overlay_img_pre]
        ).then(
            fn=lambda: generate_predicted_overlay_post([]),
            outputs=[pred_overlay_img]
        ).then(
            fn=lambda: generate_predicted_overlay_mask([]),
            outputs=[pred_overlay_img_mask]
        ).then(
            fn=lambda: generate_predicted_overlay_mask_post([]),
            outputs=[pred_overlay_img_mask_post]
        )

        pred_overlay_selection_pre.change(
            fn=generate_predicted_overlay_pre,
            inputs=[pred_overlay_selection_pre],
            outputs=[pred_overlay_img_pre]
        )

        pred_overlay_selection.change(
            fn=generate_predicted_overlay_post,
            inputs=[pred_overlay_selection],
            outputs=[pred_overlay_img]
        )

        svg_gallery_pre.select(
            fn=select_prediction_from_gallery_pre,
            inputs=[],
            outputs=[pred_overlay_selection_pre]
        )

        svg_gallery.select(
            fn=select_prediction_from_gallery_post,
            inputs=[],
            outputs=[pred_overlay_selection]
        )
        pred_overlay_selection_mask.change(generate_predicted_overlay_mask, inputs=[pred_overlay_selection_mask],
                                           outputs=[pred_overlay_img_mask])
        pred_overlay_selection_mask_post.change(generate_predicted_overlay_mask_post,
                                                inputs=[pred_overlay_selection_mask_post],
                                                outputs=[pred_overlay_img_mask_post])

        svg_gallery_mask.select(
            fn=select_prediction_from_gallery_mask,
            inputs=[],
            outputs=[pred_overlay_selection_mask]
        )

        svg_gallery_mask_post.select(
            fn=select_prediction_from_gallery_mask_post,
            inputs=[],
            outputs=[pred_overlay_selection_mask_post]
        )

        run_stitch_btn.click(
            fn=run_stitching_from_prediction,
            inputs=[threshold_input, use_hsv_checkbox],
            outputs=[csv_list_output, svg_viewer, csv_status]
        )

        download_log_btn.click(fn=save_session_log, outputs=[log_file_output])

    return demo