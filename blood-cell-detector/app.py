import gradio as gr
import requests
import base64
from PIL import Image
import io

API_URL = "http://localhost:8000/predict"

def analyze_image(image):
    # Convert Gradio image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    # Call the API
    response = requests.post(
        API_URL,
        files={"file": ("image.jpg", img_bytes, "image/jpeg")}
    )

    if response.status_code != 200:
        return None, f"API error: {response.text}"

    data = response.json()

    # Decode annotated image
    img_data = base64.b64decode(data["annotated_image"])
    annotated = Image.open(io.BytesIO(img_data))

    # Format summary
    summary = f"**Total cells detected: {data['total_cells']}**\n"
    summary += f"Mean confidence: {data['mean_confidence']:.1%}\n\n"
    summary += "**Count per class:**\n"
    for cls_name, count in data["counts_per_class"].items():
        pct = count / data["total_cells"] * 100
        summary += f"- {cls_name}: {count} ({pct:.1f}%)\n"

    return annotated, summary

with gr.Blocks(title="Blood Cell Detector") as demo:
    gr.Markdown("# 🔬 Blood Cell Detector")
    gr.Markdown("Upload a blood smear image to detect and classify blood cells.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input image")
            analyze_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Detections")
            output_text = gr.Markdown(label="Results")

    analyze_btn.click(
        analyze_image,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)