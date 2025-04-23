import yt_dlp


def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'quiet': True,
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        return info_dict['url']

