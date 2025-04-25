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

# Assume IrisPipeline and processing functions are imported/defined here
# from your_module import IrisPipeline, daugman_normalization_modified, enhanced_iris_encoder

app.layout = dbc.Container([
    html.H1("Iris Recognition System", className="mb-4 text-center"),
    
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select an Eye Image')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    
    dbc.Row([
        dbc.Col([
            html.Div(id='original-image', className="mb-4")
        ], width=6),
        
        dbc.Col([
            html.Div(id='iris-code', className="mb-4")
        ], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id='processing-status')
        ], width=12)
    ])
], fluid=True)

def parse_image(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))
    
    # Ensure image is in RGB mode if it's not already
    if image.mode == 'RGB':
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    elif image.mode == 'L':  # Already grayscale
        gray = np.array(image)
    else:
        # Convert everything else (e.g., RGBA) to RGB first
        image = image.convert('RGB')
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Get the path of the uploaded image
    
    return gray

def numpy_to_base64(image_array):
    # Convert to uint8 if needed
    if image_array.dtype != np.uint8:
        image_array = (255 * (image_array / np.max(image_array))).astype(np.uint8)

    # Convert grayscale or RGB image to PIL image
    if image_array.ndim == 2:
        pil_img = Image.fromarray(image_array, mode='L')
    elif image_array.ndim == 3:
        pil_img = Image.fromarray(image_array)
    else:
        raise ValueError("Unsupported image shape")

    # Save PIL image to bytes buffer
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")  # ← this needs a buffer, which we're giving
    buff.seek(0)  # Move to start of the buffer

    # Encode the buffer contents to base64
    img_base64 = base64.b64encode(buff.read()).decode("utf-8")
    return img_base64


@callback(
    [Output('original-image', 'children'),
     Output('iris-code', 'children'),
     Output('processing-status', 'children')],
    [Input('upload-image', 'contents')],
    prevent_initial_call=True
)
def process_image(contents):
    if not contents:
        return dash.no_update, dash.no_update, "No image uploaded"
    
    try:
        # Process image
        gray_image = parse_image(contents)

        # Supply path instead of image
        pipeline = IrisPipeline(gray_image)
        pipeline.run_pipeline()
        results = pipeline.get_results()
        
        # Normalize iris region
        normalized = daugman_normalization_modified(
            gray_image,
            results["pupil_center"],
            results["pupil_radius"],
            results["iris_radius"]
        )
        
        # Generate iris code
        iris_code = enhanced_iris_encoder(normalized)

        # Create visualizations
        original_img = numpy_to_base64(gray_image)
        
        # Scale up the iris code for better resolution
        scale_factor = 10  # Increase resolution by scaling
        iris_code_resized = cv2.resize(
            (iris_code * 255).astype(np.uint8),
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_NEAREST
        )
        code_img = numpy_to_base64(iris_code_resized)
        
        return [
            html.Img(src=f'data:image/png;base64,{original_img}',
                style={'width': '100%', 'border': '1px solid #ddd'}),
            html.Img(src=f'data:image/png;base64,{code_img}',
                style={'width': '100%', 'border': '1px solid #ddd'}),
            dbc.Alert("Processing completed successfully!", color="success")
        ]
    
    except Exception as e:
        return (
            dash.no_update,
            dash.no_update,
            dbc.Alert(f"Error processing image: {str(e)}", color="danger")
        )

if __name__ == '__main__':
    app.run_server(debug=True)