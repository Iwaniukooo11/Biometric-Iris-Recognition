import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import base64
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pipeline import IrisPipeline
from encoding import enhanced_iris_encoder
from normalization import daugman_normalization_modified

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Iris Recognition System", className="mb-4 text-center"),

    dbc.Row([
        dbc.Col([
            html.H4("Upload Eye Image", className="text-center"),
            dcc.Upload(
                id='upload-image',
                children=html.Div(id='upload-area', children=[
                    'Drag and Drop or ',
                    html.A('Select an Eye Image')
                ]),
                style={
                    'width': '100%',
                    'height': '500px',
                    'lineHeight': '300px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '10px',
                    'textAlign': 'center',
                    'margin': '10px',
                    'backgroundColor': '#f9f9f9',
                    'overflow': 'hidden'
                },
                multiple=False
            )
        ], width=6),

        dbc.Col([
            html.H4("Generated Iris Code", className="text-center"),
            html.Div(id='iris-code', style={
                'border': '1px solid #ddd',
                'borderRadius': '10px',
                'padding': '10px',
                'textAlign': 'center',
                'backgroundColor': '#f9f9f9',
                'minHeight': '500px',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center'
            }),
            html.Div([
                html.Button("Save Code Plot", id="save-plot-btn", className="btn btn-primary mt-3"),
                html.Button("Save Code as .txt", id="save-code-btn", className="btn btn-secondary mt-3 ml-2"),
                dcc.Download(id="download-plot"),
                dcc.Download(id="download-code")
            ], className="text-center mt-3")
        ], width=6)
    ]),

    dbc.Row([
        dbc.Col([
            html.Div(id='processing-status', className="mt-4")
        ], width=12)
    ])
], fluid=True)

def parse_image(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))
    
    if image.mode == 'RGB':
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    elif image.mode == 'L':
        gray = np.array(image)
    else:
        image = image.convert('RGB')
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    return gray

def numpy_to_base64(image_array):
    if image_array.dtype != np.uint8:
        image_array = (255 * (image_array / np.max(image_array))).astype(np.uint8)

    if image_array.ndim == 2:
        pil_img = Image.fromarray(image_array, mode='L')
    elif image_array.ndim == 3:
        pil_img = Image.fromarray(image_array)
    else:
        raise ValueError("Unsupported image shape")

    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    buff.seek(0)

    img_base64 = base64.b64encode(buff.read()).decode("utf-8")
    return img_base64

@callback(
    [Output('upload-area', 'children'),
     Output('iris-code', 'children'),
     Output('processing-status', 'children')],
    [Input('upload-image', 'contents')],
    prevent_initial_call=True
)
def process_image(contents):
    if not contents:
        return dash.no_update, dash.no_update, "No image uploaded"
    
    try:
        gray_image = parse_image(contents)

        pipeline = IrisPipeline(gray_image)
        pipeline.run_pipeline()
        results = pipeline.get_results()
        
        normalized = daugman_normalization_modified(
            gray_image,
            results["pupil_center"],
            results["pupil_radius"],
            results["iris_radius"]
        )
        
        iris_code = enhanced_iris_encoder(normalized)

        original_img = numpy_to_base64(gray_image)
        
        scale_factor = 10
        iris_code_resized = cv2.resize(
            (iris_code * 255).astype(np.uint8),
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_NEAREST
        )
        code_img = numpy_to_base64(iris_code_resized)
        
        global saved_iris_code  # Save the iris code globally for download callbacks
        saved_iris_code = iris_code

        return [
            html.Img(
                src=f'data:image/png;base64,{original_img}',
                style={
                    'maxWidth': '100%',
                    'maxHeight': '100%',
                    'objectFit': 'contain',
                    'margin': 'auto',
                    'display': 'block',
                    'borderRadius': '10px'
                }
            ),
            html.Img(
                src=f'data:image/png;base64,{code_img}',
                style={
                    'maxWidth': '100%',
                    'maxHeight': '100%',
                    'objectFit': 'contain',
                    'margin': 'auto',
                    'display': 'block',
                    'borderRadius': '10px'
                }
            ),
            dbc.Alert("Processing completed successfully!", color="success")
        ]
    
    except Exception as e:
        return (
            dash.no_update,
            dash.no_update,
            dbc.Alert(f"Error processing image: {str(e)}", color="danger")
        )

@callback(
    Output("download-plot", "data"),
    Input("save-plot-btn", "n_clicks"),
    prevent_initial_call=True
)
def save_plot(n_clicks):
    global saved_iris_code
    if saved_iris_code is None:
        return dash.no_update

    # Generate the plot
    fig, ax = plt.subplots()
    ax.imshow(saved_iris_code, cmap="gray")
    ax.axis("off")

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    return dcc.send_bytes(buf.getvalue(), "iris_code_plot.png")

@callback(
    Output("download-code", "data"),
    Input("save-code-btn", "n_clicks"),
    prevent_initial_call=True
)
def save_code(n_clicks):
    global saved_iris_code
    if saved_iris_code is None:
        return dash.no_update

    # Convert the iris code to a list of numbers and save as text
    code_as_text = "\n".join(map(str, saved_iris_code.flatten()))
    return dcc.send_string(code_as_text, "iris_code.txt")

if __name__ == '__main__':
    app.run_server(debug=True)