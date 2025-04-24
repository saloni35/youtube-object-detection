🎥 YouTube Object Detection & Tracking using YOLO + Streamlit

******************************************************************
A web-based tool built with Streamlit that allows you to run real-time object detection and tracking on YouTube videos using YOLO models and Deep SORT.

******************************************************************
🚀 Features 

📺 Input any YouTube video URL

🧠 Detect objects using different YOLO models and compare the results

🔁 Track detected objects using Deep SORT

🖼️ Display annotated frames in browser

⚙️ Modular structure (detection, tracking, utils, UI)

******************************************************************
🖥️ Demo

Paste a YouTube URL → Click "Run Detection" → Watch object detection and tracking in action!

******************************************************************
<pre lang="md"><code>
🧱 Project Structure

📁 youtube-object-detection/
│
├── main_ui.py                   # Streamlit app (UI only)
├── object_detection/
    └── detection.py # Core logic: detection
    └── tracking.py # Core logic: detection + tracking
    └── compare_yolo_models_detection # Core logic: comparison of detection using different YOLO versions
├── tracker/
│   └── deep_sort_tracker.py    # Deep SORT tracker
├── utils/
│   └── youtube_utils.py        # YouTube stream URL resolver
├── requirements.txt  # Environment dependencies
└── README.md

</code></pre>

******************************************************************

🔧 Setup Instructions
1. Clone the repo

```
git clone https://github.com/your-username/youtube-object-detection.git
cd youtube-object-detection
```

2. Install dependencies

If you're using pip:
`pip install -r requirements.txt`

If you're using Pipenv:
`pipenv install`

3. Run the app

`streamlit run main_ui.py`

******************************************************************

📦 Requirements
- torch

- ultralytics

- opencv-python

- yt-dlp

- Pillow

- streamlit

- deep_sort_realtime

All dependencies are listed in requirements.txt.

******************************************************************

🔁 Object Tracking Options
Currently supports:

✅ Deep SORT

Coming soon (easy to plug in):

⚡ SORT

🔁 ByteTrack

🤖 OC-SORT

💡 StrongSORT

******************************************************************

📚 Acknowledgements
- YOLOv5/YOLOv8 - Ultralytics

- Deep SORT Realtime

- yt-dlp

******************************************************************

💡 Future Improvements
- Allow tracker selection from dropdown

- Use full video stream (not just some frames)

- Display FPS and performance metrics

- Export results as video or annotated images

******************************************************************

🙌 Contributing
Pull requests are welcome! Please open an issue to discuss before major changes.

******************************************************************

📜 License
MIT License – Use it freely for educational and personal projects.