<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
</head>
<body>
  <h1>YOLOv8 Object Detection App</h1>
  
  <div class="section">
    <h2 class="section-title">Description</h2>
    <p>
      This is a web application built with Flask that performs object detection using YOLOv8 model. It allows you to upload images or videos, or use the webcam for real-time object detection. The detected objects are labeled with bounding boxes and class names.
    </p>
  </div>
  
  <div class="section">
    <h2 class="section-title">Installation</h2>
    <div class="code-block">
      <code>$ git clone https://github.com/egypt-metro/egypt-metro-ai.git</code><br>
      <code>$ cd egypt-metro-ai</code><br>
      <code>$ pip install -r requirements.txt</code>
    </div>
  </div>
  
  <div class="section">
    <h2 class="section-title">Usage</h2>
    <p>Follow the steps below to run the application:</p>
    <ol>
      <li>Make sure you have Python and pip installed.</li>
      <li>Install the required dependencies by running the following command in the project directory:</li>
      <div class="code-block">
        <code>$ pip install -r requirements.txt</code>
      </div>
      <li>Run the application using the following command:</li>
      <div class="code-block">
        <code>$ python main.py</code>
      </div>
      <li>Open your web browser and visit <code>http://localhost:5000</code>.</li>
      <li>Upload an image or video file, or use the webcam for real-time object detection.</li>
      <li>View the object detection results on the web page.</li>
    </ol>
  </div>
  
  <div class="section">
    <h2 class="section-title">File Structure</h2>
    <ul>
      <li><strong>maln.py:</strong> The main Flask application file.</li>
      <li><strong>infer.py:</strong> Contains functions for running YOLOv8 object detection.</li>
      <li><strong>templates:</strong> Contains HTML templates for rendering the web pages.

<li><strong>static/web_images:</strong> Contains static images used in the web application.</li>
<li><strong>yolo_assets:</strong> Contains the YOLOv8 model, class names file, and output directory for detections.</li>
<li><strong>README.md:</strong> The README file with instructions and information about the project.</li>
<li><strong>requirements.txt:</strong> Lists the required Python packages and their versions.</li>
</ul>
</div>
<div class="section">
<h2 class="section-title">Dependencies</h2>
<p>The project relies on the following dependencies:</p>
<div class="code-block">
<code>Flask==2.3.2</code><br>
<code>Flask-WTF==1.1.1</code><br>
<code>opencv-python==4.7.0.72</code><br>
<code>ultralytics==8.0.99</code><br>
</div>
</div>
<div class="section">
<h2 class="section-title">Contributing</h2>
<p>Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to create an issue or submit a pull request.</p>
</div>
<div class="section">
<h2 class="section-title">License</h2>
<p>This project is licensed under the MIT License. See the <code>LICENSE</code> file for more information.</p>
</div>
<div class="section">
<h2 class="section-title">Acknowledgments</h2>
<p>Special thanks to the creators of YOLOv8 and Flask for their excellent frameworks.</p>
</div>
<div class="section">
<h2 class="section-title">Contact</h2>
<p>If you have any questions or inquiries, please contact me at your-email@example.com.</p>
</div>
</body>
</html>
