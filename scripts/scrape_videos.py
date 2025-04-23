import requests
import re
import os
import yt_dlp
import cv2
import mediapipe as mp
import sys
import logging
from core.config import CONFIG
from core.db_utils import store_video_metadata
from utils.video_utils import is_video_good

logging.basicConfig(level=logging.INFO)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def scrape_youtube_search(query, target_count=35):
    all_urls = []
    page_token = None
    while len(all_urls) < target_count * 2:
        try:
            url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
            if page_token:
                url += f"&sp={page_token}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            video_ids = re.findall(r'(?:/watch?v=|/shorts/)([a-zA-Z0-9_-]{11})', response.text)
            new_urls = [f"https://www.youtube.com/watch?v={vid}" for vid in set(video_ids) if vid not in [u.split('=')[-1] for u in all_urls]]
            all_urls.extend(new_urls)
            next_page = re.search(r'"continuation":"([^"]+)"', response.text)
            page_token = next_page.group(1) if next_page else None
            if not page_token:
                break
        except Exception as e:
            logging.error(f"Error scraping: {e}")
            break
    return all_urls[:target_count * 2]

def download_videos(urls, output_dir, target_count=35, action_type="fast", db_path="~/bowliverse_ai/bowliverse.db"):
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {"format": "bestvideo[height<=720]", "outtmpl": f"{output_dir}/{action_type}_%(id)s.%(ext)s", "quiet": True}
    downloaded = 0
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in urls[:target_count * 2]:
            if downloaded >= target_count:
                break
            video_id = url.split('v=')[-1]
            video_path = f"{output_dir}/{action_type}_{video_id}.mp4"
            try:
                ydl.download([url])
                if is_video_good(video_path, action_type):
                    store_video_metadata(db_path, video_id, url, action_type)
                    downloaded += 1
                    logging.info(f"Accepted {video_id}")
                else:
                    os.remove(video_path)
            except Exception as e:
                logging.error(f"Error downloading {video_id}: {e}")

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "fast"
    query = f"Complete {action} bowling action slow motion"
    urls = scrape_youtube_search(query)
    download_videos(urls, "../videos", action_type=action)
