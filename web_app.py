"""
Web Interface for DCGAN Image Generation

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Flask web application for generating images using trained DCGAN models.
"""

from flask import Flask, render_template, send_file, request, jsonify
import torch
import io
import base64
from PIL import Image
import os
from dcgan_model import Generator
import config

app = Flask(__name__)

# Global variables
generator = None
device = None


def load_generator(checkpoint_path):
    """
    Load trained generator
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    global generator, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generator = Generator(
        nz=config.NZ,
        ngf=config.NGF,
        nc=config.NC,
        image_size=config.IMAGE_SIZE
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.eval()
    
    print(f"Generator loaded from {checkpoint_path}")


@app.route('/')
def index():
    """
    Main page
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate images
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    if generator is None:
        return jsonify({'error': 'Generator not loaded'}), 400
    
    try:
        num_images = int(request.json.get('num_images', 1))
        num_images = min(max(num_images, 1), 16)  # Limit to 16
        
        # Generate images
        noise = torch.randn(num_images, config.NZ, 1, 1, device=device)
        
        with torch.no_grad():
            images = generator(noise).cpu()
        
        # Convert to PIL images
        results = []
        for i in range(num_images):
            img = images[i]
            # Denormalize from [-1, 1] to [0, 1]
            img = (img + 1) / 2
            img = torch.clamp(img, 0, 1)
            
            # Convert to PIL
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype('uint8')
            pil_img = Image.fromarray(img_np)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            results.append(f'data:image/png;base64,{img_str}')
        
        return jsonify({'images': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'generator_loaded': generator is not None,
        'device': str(device) if device else None
    })


def create_web_templates():
    """
    Create HTML template for web interface
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    os.makedirs('templates', exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DCGAN Image Generation - RSK World</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }
        
        h1 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #666;
            margin-bottom: 30px;
        }
        
        .controls {
            margin-bottom: 30px;
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .controls input {
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        
        .controls button {
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .controls button:hover {
            transform: translateY(-2px);
        }
        
        .controls button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .image-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 10px;
            text-align: center;
        }
        
        .image-card img {
            width: 100%;
            height: auto;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        
        .author-info {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
            font-size: 14px;
        }
        
        .author-info a {
            color: #667eea;
            text-decoration: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 DCGAN Image Generation</h1>
        <p class="subtitle">Generate realistic images using Deep Convolutional GAN</p>
        
        <div class="controls">
            <label>Number of Images:</label>
            <input type="number" id="numImages" value="4" min="1" max="16">
            <button onclick="generateImages()" id="generateBtn">Generate Images</button>
        </div>
        
        <div id="gallery" class="gallery"></div>
        
        <div class="author-info">
            <p><strong>Author:</strong> RSK World | 
               <a href="https://rskworld.in" target="_blank">Website</a> | 
               <a href="mailto:help@rskworld.in">Email</a> | 
               Phone: +91 93305 39277</p>
        </div>
    </div>
    
    <script>
        async function generateImages() {
            const btn = document.getElementById('generateBtn');
            const gallery = document.getElementById('gallery');
            const numImages = parseInt(document.getElementById('numImages').value);
            
            btn.disabled = true;
            btn.textContent = 'Generating...';
            gallery.innerHTML = '<div class="loading">Generating images, please wait...</div>';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ num_images: numImages })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    gallery.innerHTML = `<div class="loading">Error: ${data.error}</div>`;
                } else {
                    gallery.innerHTML = '';
                    data.images.forEach((imgSrc, index) => {
                        const card = document.createElement('div');
                        card.className = 'image-card';
                        card.innerHTML = `
                            <img src="${imgSrc}" alt="Generated Image ${index + 1}">
                            <p>Image ${index + 1}</p>
                        `;
                        gallery.appendChild(card);
                    });
                }
            } catch (error) {
                gallery.innerHTML = `<div class="loading">Error: ${error.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Generate Images';
            }
        }
        
        // Generate on page load
        window.onload = function() {
            generateImages();
        };
    </script>
</body>
</html>'''
    
    with open('templates/index.html', 'w') as f:
        f.write(html_content)
    
    print("Web template created in templates/index.html")


if __name__ == '__main__':
    # Create template
    create_web_templates()
    
    # Load generator
    checkpoint_path = 'checkpoints/final_generator.pth'
    if os.path.exists(checkpoint_path):
        load_generator(checkpoint_path)
        print("Starting web server...")
        print("Open http://localhost:5000 in your browser")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train a model first or provide a valid checkpoint path.")

