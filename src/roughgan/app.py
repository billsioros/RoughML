from pathlib import Path

import gradio as gr
import plotly.express as px
import torch

from roughgan.models.cnn import CNNGenerator


def predict():
    load_state = Path.cwd() / "models" / "CNNGenerator.pt"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CNNGenerator.from_state(load_state, device=device)

    model.eval()
    with torch.no_grad():
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        bert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=56,
            truncation=True,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        outputs = bert(input_ids=input_ids, attention_mask=attention_mask)

        image = model(outputs.last_hidden_state)

        def as_grayscale_image(array, save_path=None):
            fig = px.imshow(array)
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)

            if save_path is None:
                fig.show()
            else:
                with save_path.open("wb") as file:
                    fig.write_image(file)

        as_grayscale_image(
            image.cpu().detach().numpy().squeeze().reshape(256, 256, 3),
            save_path=Path.cwd() / f"{text}.png",
        )


gr.Interface(
    predict,
    inputs=[
        gr.Slider(0, 1000, label="Seed", default=42),
    ],
    outputs="image",
).launch()
